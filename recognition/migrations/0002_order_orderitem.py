# Generated manually

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('recognition', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Order',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('order_no', models.CharField(max_length=32, unique=True, verbose_name='订单号')),
                ('total', models.DecimalField(decimal_places=2, max_digits=10, verbose_name='总计')),
                ('item_count', models.IntegerField(verbose_name='商品种类数')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='创建时间')),
            ],
            options={
                'verbose_name': '订单',
                'verbose_name_plural': '订单',
                'ordering': ['-created_at'],
            },
        ),
        migrations.CreateModel(
            name='OrderItem',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('product_name', models.CharField(max_length=100, verbose_name='商品名称')),
                ('price', models.DecimalField(decimal_places=2, max_digits=10, verbose_name='单价')),
                ('quantity', models.IntegerField(verbose_name='数量')),
                ('subtotal', models.DecimalField(decimal_places=2, max_digits=10, verbose_name='小计')),
                ('order', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='items', to='recognition.order', verbose_name='订单')),
            ],
            options={
                'verbose_name': '订单明细',
                'verbose_name_plural': '订单明细',
            },
        ),
    ]
