import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from django.conf import settings

MIN_CONFIDENCE = 0.6


class ProductPredictor:
    def __init__(self):
        self.model = None
        self.label_names = {}
        self.label_map = {}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
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
            nn.Dropout(0.2),
            nn.Linear(self.model.last_channel, num_classes),
        )

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.loaded = True
        return True, f'模型加载成功（{num_classes}个类别）'

    def predict(self, image_path):
        if not self.loaded:
            ok, msg = self.load_model()
            if not ok:
                return {'success': False, 'message': msg}

        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = probabilities.max(1)

            confidence = confidence.item()
            predicted_idx = predicted_idx.item()
            product_name = self.label_names.get(predicted_idx, '未知商品')

            if confidence < MIN_CONFIDENCE:
                all_probs = probabilities[0].cpu().numpy()
                all_results = []
                for idx, prob in enumerate(all_probs):
                    all_results.append({
                        'name': self.label_names.get(idx, f'类别{idx}'),
                        'confidence': round(float(prob), 4),
                    })
                all_results.sort(key=lambda x: x['confidence'], reverse=True)

                return {
                    'success': False,
                    'message': f'置信度过低（{confidence:.1%}），无法确认商品。请调整角度或距离后重试。',
                    'confidence': round(confidence, 4),
                    'top_guess': product_name,
                    'all_results': all_results[:5],
                }

            all_probs = probabilities[0].cpu().numpy()
            all_results = []
            for idx, prob in enumerate(all_probs):
                all_results.append({
                    'name': self.label_names.get(idx, f'类别{idx}'),
                    'confidence': round(float(prob), 4),
                })
            all_results.sort(key=lambda x: x['confidence'], reverse=True)

            return {
                'success': True,
                'product_name': product_name,
                'confidence': round(confidence, 4),
                'all_results': all_results[:5],
            }

        except Exception as e:
            return {'success': False, 'message': f'预测失败: {str(e)}'}


_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = ProductPredictor()
    return _predictor


def predict_image(image_path):
    return get_predictor().predict(image_path)
