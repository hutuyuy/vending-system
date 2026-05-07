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

from PIL import Image
from django.conf import settings

logger = logging.getLogger('recognition')


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


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


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
    return image_paths, labels, label_map, label_names


def train_model(epochs=30, batch_size=8, learning_rate=0.001, patience=5):
    image_paths, labels, label_map, label_names = collect_data()

    if len(image_paths) == 0:
        return {'success': False, 'message': '没有找到训练数据！请先在后台为商品上传图片。'}
    if len(label_names) < 2:
        return {'success': False, 'message': f'至少需要2种商品才能训练（当前只有{len(label_names)}种）。'}

    train_transform, val_transform = get_transforms()

    # 分别创建训练集和验证集，验证集使用 val_transform（无数据增强）
    full_indices = list(range(len(image_paths)))
    train_size = int(0.8 * len(full_indices))
    val_size = len(full_indices) - train_size
    train_indices, val_indices = torch.utils.data.random_split(full_indices, [train_size, val_size])

    train_dataset = ProductDataset(
        [image_paths[i] for i in train_indices],
        [labels[i] for i in train_indices],
        train_transform,
    )
    val_dataset = ProductDataset(
        [image_paths[i] for i in val_indices],
        [labels[i] for i in val_indices],
        val_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    for param in model.features.parameters():
        param.requires_grad = False

    num_classes = len(label_names)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,)

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
        current_lr = optimizer.param_groups[0]['lr']

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

        logger.info(f'Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Train: {train_acc:.1f}% Val: {val_acc:.1f}% LR: {current_lr:.6f} Time: {epoch_time}s')

        if no_improve >= patience:
            logger.info(f'Early stopping at epoch {epoch+1}，{patience}轮无提升')
            break

    # 保存训练历史
    save_history(history)

    # 重置预测器单例
    import recognition.ml.predictor as pred_module
    pred_module._predictor = None

    return {
        'success': True,
        'message': f'训练完成！最优验证准确率：{best_acc:.1f}%',
        'epochs': len(history['train_loss']),
        'train_samples': train_size,
        'val_samples': val_size,
        'num_classes': num_classes,
        'best_accuracy': round(best_acc, 1),
        'label_names': {str(k): v for k, v in label_names.items()},
        'early_stopped': no_improve >= patience,
    }


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
    history_path = os.path.join(settings.BASE_DIR, 'recognition', 'ml', 'saved_models', 'train_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None
