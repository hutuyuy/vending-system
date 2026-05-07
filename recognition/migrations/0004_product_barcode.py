from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('recognition', '0003_product_is_active_product_low_stock_threshold_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='product',
            name='barcode',
            field=models.CharField(
                blank=True, default='',
                help_text='商品条形码或二维码内容，用于扫码识别',
                max_length=50, verbose_name='条形码',
            ),
        ),
    ]
