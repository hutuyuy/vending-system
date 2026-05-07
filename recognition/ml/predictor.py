import os
import json
import logging

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from PIL import Image
from django.conf import settings

logger = logging.getLogger('recognition')

# ========================================================
# 提高置信度阈值：宁可说"不认识"也别认错
# ========================================================
MIN_CONFIDENCE = 0.50


class ProductPredictor:
    def __init__(self):
        if not HAS_TORCH:
            self.loaded = False
            return
        self.model = None
        self.label_names = {}
        self.label_map = {}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # 测试时增强 (TTA)：多角度预测取平均
        self.tta_transforms = [
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        ]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loaded = False

    def load_model(self):
        model_dir = os.path.join(settings.BASE_DIR, 'recognition', 'ml', 'saved_models')
        model_path = os.path.join(model_dir, 'product_model.pth')
        meta_path = os.path.join(model_dir, 'model_meta.json')

        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            return False, '模型文件不存在，请先训练模型'

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        self.label_map = meta['label_map']
        self.label_names = {int(k): v for k, v in meta['label_names'].items()}
        num_classes = meta['num_classes']

        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.last_channel, num_classes),
        )

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.loaded = True
        return True, f'模型加载成功（{num_classes}个类别）'

    def predict_single(self, image):
        """单张图片预测（带 TTA）"""
        # 主预测
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)

        # TTA：多尺度 + 翻转取平均
        tta_probs = [probs]
        for t in self.tta_transforms[1:]:
            aug_input = t(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                aug_out = self.model(aug_input)
                tta_probs.append(torch.softmax(aug_out, dim=1))

        avg_probs = torch.stack(tta_probs).mean(dim=0)
        return avg_probs

    def predict(self, image_path):
        """单张图片识别"""
        if not self.loaded:
            ok, msg = self.load_model()
            if not ok:
                return {'success': False, 'message': msg}

        try:
            image = Image.open(image_path).convert('RGB')
            avg_probs = self.predict_single(image)
            return self._build_result(avg_probs)

        except Exception as e:
            logger.exception(f'预测失败: {e}')
            return {'success': False, 'message': f'预测失败: {str(e)}'}

    def predict_multiple(self, image_paths):
        """多图投票识别 — 拿多张图综合判断"""
        if not self.loaded:
            ok, msg = self.load_model()
            if not ok:
                return {'success': False, 'message': msg}

        if not image_paths:
            return {'success': False, 'message': '没有图片'}

        try:
            all_probs = []
            for path in image_paths:
                if not os.path.exists(path):
                    continue
                image = Image.open(path).convert('RGB')
                probs = self.predict_single(image)
                all_probs.append(probs)

            if not all_probs:
                return {'success': False, 'message': '没有有效的图片'}

            # 多图概率取平均
            avg_probs = torch.stack(all_probs).mean(dim=0)
            result = self._build_result(avg_probs)
            result['image_count'] = len(all_probs)
            return result

        except Exception as e:
            logger.exception(f'多图预测失败: {e}')
            return {'success': False, 'message': f'预测失败: {str(e)}'}

    def _build_result(self, avg_probs):
        """从概率张量构建返回结果"""
        confidence, predicted_idx = avg_probs.max(1)
        confidence = confidence.item()
        predicted_idx = predicted_idx.item()
        product_name = self.label_names.get(predicted_idx, '未知商品')

        all_probs_np = avg_probs[0].cpu().numpy()
        all_results = []
        for idx, prob in enumerate(all_probs_np):
            all_results.append({
                'name': self.label_names.get(idx, f'类别{idx}'),
                'confidence': round(float(prob), 4),
            })
        all_results.sort(key=lambda x: x['confidence'], reverse=True)

        if confidence < MIN_CONFIDENCE:
            return {
                'success': False,
                'message': (
                    f'置信度过低（{confidence:.1%}），无法确认商品。'
                    f'请调整角度或距离后重试。'
                ),
                'confidence': round(confidence, 4),
                'top_guess': product_name,
                'all_results': all_results[:5],
            }

        return {
            'success': True,
            'product_name': product_name,
            'confidence': round(confidence, 4),
            'all_results': all_results[:5],
        }


# ============================================================
# 单例
# ============================================================

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = ProductPredictor()
    return _predictor


def predict_image(image_path):
    """兼容旧接口"""
    return get_predictor().predict(image_path)


def predict_images(image_paths):
    """多图识别新接口"""
    return get_predictor().predict_multiple(image_paths)
