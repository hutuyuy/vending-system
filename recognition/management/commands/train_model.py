"""
商品图像分类模型训练 - 管理命令
直接复用 recognition.ml.trainer 模块，避免代码重复
"""
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = '训练商品识别模型（MobileNetV2 迁移学习）'

    def add_arguments(self, parser):
        parser.add_argument('--epochs', type=int, default=30, help='训练轮数（默认30）')
        parser.add_argument('--batch-size', type=int, default=8, help='批次大小（默认8）')
        parser.add_argument('--lr', type=float, default=0.001, help='学习率（默认0.001）')
        parser.add_argument('--patience', type=int, default=5, help='早停轮数（默认5）')

    def handle(self, *args, **options):
        from recognition.ml.trainer import train_model

        self.stdout.write('🚀 开始训练...')
        result = train_model(
            epochs=options['epochs'],
            batch_size=options['batch_size'],
            learning_rate=options['lr'],
            patience=options['patience'],
        )

        if result['success']:
            self.stdout.write(self.style.SUCCESS(f"✅ {result['message']}"))
            self.stdout.write(f"   样本: {result['train_samples']}+{result['val_samples']}")
            self.stdout.write(f"   类别: {result['num_classes']}")
            self.stdout.write(f"   准确率: {result['best_accuracy']}%")
        else:
            self.stdout.write(self.style.ERROR(f"❌ {result['message']}"))
