import os
import json
import time
import logging

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms, models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from PIL import Image, ImageFilter, ImageEnhance
from django.conf import settings
from sklearn.model_selection import train_test_split

logger = logging.getLogger('recognition')


# ============================================================
#  数据集
# ============================================================

class ProductDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ============================================================
#  数据增强 — 比原来更贴近真实售货场景
# ============================================================

def get_transforms():
    """训练时增强更激进，模拟真实货架环境"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


# ============================================================
#  数据收集 + 每张图复制增强版本扩充小样本
# ============================================================

def collect_data():
    from recognition.models import Product, ProductImage
    image_paths = []
    labels = []
    label_map = {}
    label_names = {}
    products = Product.objects.all()
    for idx, product in enumerate(products):
        images = ProductImage.objects.filter(product=product)
        if not images.exists():
            continue
        label_map[product.id] = idx
        label_names[idx] = product.name
        for img in images:
            full_path = os.path.join(settings.MEDIA_ROOT, str(img.image))
            if os.path.exists(full_path):
                image_paths.append(full_path)
                labels.append(idx)

    # 记录原始样本数（不包含增强副本）
    original_count = len(image_paths)
    logger.info(f'原始训练样本: {original_count} 张, {len(label_names)} 个类别')

    # 如果每个类别样本少于 20 张，用轻量离线增强扩充
    from collections import Counter
    label_counts = Counter(labels)
    augmented_paths = list(image_paths)
    augmented_labels = list(labels)

    offline_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    ])

    temp_dir = os.path.join(settings.BASE_DIR, 'recognition', 'ml', 'temp_aug')
    os.makedirs(temp_dir, exist_ok=True)

    for label_id, count in label_counts.items():
        if count < 20:
            need = 20 - count
            class_paths = [p for p, l in zip(image_paths, labels) if l == label_id]
            for i in range(need):
                src = class_paths[i % len(class_paths)]
                # 生成增强副本（保存到临时目录）
                try:
                    img = Image.open(src).convert('RGB')
                    aug_img = offline_aug(img)
                    aug_path = os.path.join(temp_dir, f'aug_{label_id}_{i}.jpg')
                    aug_img.save(aug_path, 'JPEG', quality=85)
                    augmented_paths.append(aug_path)
                    augmented_labels.append(label_id)
                except Exception as e:
                    logger.warning(f'离线增强失败 {src}: {e}')

    logger.info('扩充后训练样本: %d 张', len(augmented_paths))
    return augmented_paths, augmented_labels, label_map, label_names


# ============================================================
#  训练主函数
# ============================================================

def train_model(epochs=30, batch_size=8, learning_rate=0.001, patience=5):
    image_paths, labels, label_map, label_names = collect_data()

    if len(image_paths) == 0:
        return {'success': False, 'message': '没有找到训练数据！请先在后台为商品上传图片。'}
    if len(label_names) < 2:
        return {'success': False, 'message': f'至少需要2种商品才能训练（当前只有{len(label_names)}种）。'}

    train_transform, val_transform = get_transforms()

    # 分层采样划分训练集和验证集
    full_indices = list(range(len(image_paths)))
    train_idx, val_idx = train_test_split(
        full_indices, test_size=0.2,
        stratify=[labels[i] for i in full_indices],
        random_state=42
    )
    train_dataset = ProductDataset(
        [image_paths[i] for i in train_idx],
        [labels[i] for i in train_idx],
        train_transform,
    )
    val_dataset = ProductDataset(
        [image_paths[i] for i in val_idx],
        [labels[i] for i in val_idx],
        val_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # ========================================================
    # 关键改动：解冻最后 50% 的特征层，让模型学到商品特有特征
    # ========================================================
    total_layers = len(model.features)
    freeze_until = int(total_layers * 0.5)  # 前 50% 冻结，后 50% 可训练

    for i, param in enumerate(model.features.parameters()):
        if i < freeze_until:
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'可训练参数: {trainable:,} / {total_params:,} ({trainable/total_params*100:.1f}%)')

    num_classes = len(label_names)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),   # 稍微提高 dropout 防过拟合
        nn.Linear(model.last_channel, num_classes),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 分层学习率：特征层小学习率，分类器大学习率
    feature_params = [p for p in model.features.parameters() if p.requires_grad]
    classifier_params = model.classifier.parameters()

    optimizer = optim.Adam([
        {'params': feature_params, 'lr': learning_rate * 0.1},   # 特征层 10x 小学习率
        {'params': classifier_params, 'lr': learning_rate},       # 分类器正常学习率
    ], weight_decay=1e-4)  # 加 L2 正则

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3,
    )

    best_acc = 0.0
    no_improve = 0
    history = {
        'train_loss': [], 'train_acc': [], 'val_acc': [],
        'lr': [], 'epoch_time': [],
        'confusion_matrix': [],
        'label_names': {str(k): v for k, v in label_names.items()},
    }

    for epoch in range(epochs):
        epoch_start = time.time()

        # 训练
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, lbls in train_loader:
            images, lbls = images.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += lbls.size(0)
            correct += predicted.eq(lbls).sum().item()

        train_acc = 100.0 * correct / total
        avg_loss = running_loss / len(train_loader)

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, lbls in val_loader:
                images, lbls = images.to(device), lbls.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += lbls.size(0)
                val_correct += predicted.eq(lbls).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(lbls.cpu().numpy())

        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
        epoch_time = round(time.time() - epoch_start, 2)
        current_lr = optimizer.param_groups[1]['lr']  # 取分类器 lr

        # 混淆矩阵
        cm = [[0] * num_classes for _ in range(num_classes)]
        for p, l in zip(all_preds, all_labels):
            cm[l][p] += 1

        # 记录历史
        history['train_loss'].append(round(avg_loss, 4))
        history['train_acc'].append(round(train_acc, 1))
        history['val_acc'].append(round(val_acc, 1))
        history['lr'].append(current_lr)
        history['epoch_time'].append(epoch_time)
        history['confusion_matrix'].append(cm)

        # 学习率调度
        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            save_model(model, label_map, label_names, num_classes)
        else:
            no_improve += 1

        logger.info(
            f'Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} '
            f'Train: {train_acc:.1f}% Val: {val_acc:.1f}% '
            f'LR: {current_lr:.6f} Time: {epoch_time}s'
        )

        if no_improve >= patience:
            logger.info(f'Early stopping at epoch {epoch+1}，{patience}轮无提升')
            break

    # 保存训练历史
    save_history(history)

    # 重置预测器单例
    import recognition.ml.predictor as pred_module
    pred_module._predictor = None

    # 清理离线增强副本（清理 temp_aug 目录）
    temp_dir = os.path.join(settings.BASE_DIR, 'recognition', 'ml', 'temp_aug')
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info('已清理临时增强目录')

    return {
        'success': True,
        'message': f'训练完成！最优验证准确率：{best_acc:.1f}%',
        'epochs': len(history['train_loss']),
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'num_classes': num_classes,
        'best_accuracy': round(best_acc, 1),
        'label_names': {str(k): v for k, v in label_names.items()},
        'early_stopped': no_improve >= patience,
    }


# ============================================================
#  保存 / 加载
# ============================================================

def save_model(model, label_map, label_names, num_classes):
    model_dir = os.path.join(settings.BASE_DIR, 'recognition', 'ml', 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'product_model.pth')
    torch.save(model.state_dict(), model_path)
    meta_path = os.path.join(model_dir, 'model_meta.json')
    meta = {
        'num_classes': num_classes,
        'label_map': {str(k): v for k, v in label_map.items()},
        'label_names': {str(k): v for k, v in label_names.items()},
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f'模型已保存到: {model_path}')


def save_history(history):
    model_dir = os.path.join(settings.BASE_DIR, 'recognition', 'ml', 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    history_path = os.path.join(model_dir, 'train_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    logger.info(f'训练历史已保存到: {history_path}')


def load_history():
    history_path = os.path.join(
        settings.BASE_DIR, 'recognition', 'ml', 'saved_models', 'train_history.json'
    )
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None
