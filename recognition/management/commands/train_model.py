"""
商品图像分类模型训练器
使用预训练的 MobileNetV2 进行迁移学习
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from django.conf import settings


class ProductDataset(Dataset):
    """自定义商品图片数据集"""

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
    """获取图像预处理和数据增强"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.24, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((24, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.25]),
    ])

    return train_transform, val_transform


def collect_data():
    """从数据库收集商品图片数据"""
    from recognition.models import Product, ProductImage

    image_paths = []
    labels = []
    label_map = {}  # product_id -> label_index
    label_names = {}  # label_index -> product_name

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


def train_model(epochs=15, batch_size=8, learning_rate=0.01):
    """
    训练商品识别模型

    参数:
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率

    返回:
        dict: 训练结果信息
    """
    # 收集数据
    image_paths, labels, label_map, label_names = collect_data()

    if len(image_paths) == 0:
        return {
            'success': False,
            'message': '没有找到训练数据！请先在后台为商品上传图片。',
        }

    if len(label_names) < 2:
        return {
            'success': False,
            'message': f'至少需要2种商品才能训练（当前只有{len(label_names)}种）。请添加更多商品。',
        }

    # 图像变换
    train_transform, val_transform = get_transforms()

    # 创建数据集
    dataset = ProductDataset(image_paths, labels, train_transform)

    # 划分训练集和验证集（80/20）
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 使用预训练的 MobileNetV2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # 冻结前面的层，只训练最后的分类器
    for param in model.features.parameters():
        param.requires_grad = False

    # 替换分类头
    num_classes = len(label_names)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes),
    )
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # 训练循环
    best_acc = 0.0
    train_losses = []

    for epoch in range(epochs):
        # 训练阶段
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
        train_losses.append(avg_loss)

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, lbls in val_loader:
                images, lbls = images.to(device), lbls.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += lbls.size(0)
                val_correct += predicted.eq(lbls).sum().item()

        val_acc = 10.0 * val_correct / val_total if val_total > 0 else 0

        if val_acc > best_acc:
            best_acc = val_acc
            # 保存最优模型
            save_model(model, label_map, label_names, num_classes)

        print(f'Epoch [{epoch+1}/{epochs}] '
              f'Loss: {avg_loss:.4f} '
              f'Train Acc: {train_acc:.1f}% '
              f'Val Acc: {val_acc:.1f}%')

    # 训练完成后重置预测器单例，下次预测自动加载新模型
    import recognition.ml.predictor as pred_module
    pred_module._predictor = None

    return {
        'success': True,
        'message': f'训练完成！最优验证准确率：{best_acc:.1f}%',
        'epochs': epochs,
        'train_samples': train_size,
        'val_samples': val_size,
        'num_classes': num_classes,
        'best_accuracy': round(best_acc, 1),
        'label_names': label_names,
    }


def save_model(model, label_map, label_names, num_classes):
    """保存模型和标签映射"""
    model_dir = os.path.join(settings.BASE_DIR, 'recognition', 'ml', 'saved_models')
    os.makedirs(model_dir, exist_ok=True)

    # 保存模型权重
    model_path = os.path.join(model_dir, 'product_model.pth')
    torch.save(model.state_dict(), model_path)

    # 保存标签映射
    meta_path = os.path.join(model_dir, 'model_meta.json')
    meta = {
        'num_classes': num_classes,
        'label_map': {str(k): v for k, v in label_map.items()},
        'label_names': {str(k): v for k, v in label_names.items()},
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'模型已保存到: {model_path}')
