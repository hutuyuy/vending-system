from django.db import models


class Product(models.Model):
    name = models.CharField('商品名称', max_length=100)
    price = models.DecimalField('价格', max_digits=10, decimal_places=2)
    description = models.TextField('商品描述', blank=True, default='')
    created_at = models.DateTimeField('创建时间', auto_now_add=True)

    class Meta:
        verbose_name = '商品'
        verbose_name_plural = '商品'

    def __str__(self):
        return f'{self.name} - ¥{self.price}'


class ProductImage(models.Model):
    VIEW_CHOICES = [
        ('front', '正面'),
        ('back', '背面'),
        ('left', '左侧面'),
        ('right', '右侧面'),
        ('top', '顶部'),
        ('bottom', '底部'),
        ('left_front', '左前斜角'),
        ('right_front', '右前斜角'),
    ]

    product = models.ForeignKey(Product, on_delete=models.CASCADE,
                                related_name='images', verbose_name='商品')
    image = models.ImageField('图片', upload_to='products/')
    view_angle = models.CharField('视角', max_length=20, choices=VIEW_CHOICES)

    class Meta:
        verbose_name = '商品图片'
        verbose_name_plural = '商品图片'

    def __str__(self):
        return f'{self.product.name} - {self.get_view_angle_display()}'
