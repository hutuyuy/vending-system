from django.contrib import admin
from django.utils.html import format_html
from .models import Product, ProductImage, Order, OrderItem


class ProductImageInline(admin.TabularInline):
    model = ProductImage
    extra = 1
    fields = ['image', 'view_angle', 'image_preview']
    readonly_fields = ['image_preview']

    def image_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="max-height:80px; border-radius:6px;" />',
                obj.image.url
            )
        return '-'
    image_preview.short_description = '预览'


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ['name', 'price', 'image_count', 'status_badge', 'created_at']
    list_display_links = ['name']
    search_fields = ['name', 'description']
    list_filter = ['created_at']
    list_per_page = 20
    ordering = ['-created_at']
    inlines = [ProductImageInline]

    fieldsets = (
        ('基本信息', {
            'fields': ('name', 'price', 'description'),
        }),
    )

    def image_count(self, obj):
        count = obj.images.count()
        if count >= 8:
            return format_html('<span style="color:#10b981; font-weight:600;">{} 张 ✅</span>', count)
        elif count > 0:
            return format_html('<span style="color:#f59e0b; font-weight:600;">{} 张 ⚠️</span>', count)
        return format_html('<span style="color:#ef4444;">0 张 ❌</span>')
    image_count.short_description = '图片数量'

    def status_badge(self, obj):
        count = obj.images.count()
        if count >= 8:
            return format_html(
                '<span style="background:rgba(16,185,129,0.15); color:#059669; padding:3px 10px; '
                'border-radius:12px; font-size:12px; font-weight:600;">就绪</span>'
            )
        elif count > 0:
            return format_html(
                '<span style="background:rgba(245,158,11,0.15); color:#d97706; padding:3px 10px; '
                'border-radius:12px; font-size:12px; font-weight:600;">图片不足</span>'
            )
        return format_html(
            '<span style="background:rgba(239,68,68,0.15); color:#dc2626; padding:3px 10px; '
            'border-radius:12px; font-size:12px; font-weight:600;">无图片</span>'
        )
    status_badge.short_description = '状态'


@admin.register(ProductImage)
class ProductImageAdmin(admin.ModelAdmin):
    list_display = ['product', 'view_angle', 'image_preview', 'image_size']
    list_display_links = ['product']
    list_filter = ['view_angle', 'product']
    search_fields = ['product__name']
    list_per_page = 30

    def image_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="max-height:60px; border-radius:4px;" />',
                obj.image.url
            )
        return '-'
    image_preview.short_description = '预览'

    def image_size(self, obj):
        if obj.image:
            try:
                size_kb = obj.image.size / 1024
                if size_kb > 1024:
                    return f'{size_kb / 1024:.1f} MB'
                return f'{size_kb:.0f} KB'
            except (OSError, ValueError):
                return '-'
        return '-'
    image_size.short_description = '文件大小'


# 自定义 Admin 站点标题
admin.site.site_header = '🛒 智能售货系统 - 管理后台'
admin.site.site_title = '智能售货系统'
admin.site.index_title = '系统管理'


class OrderItemInline(admin.TabularInline):
    model = OrderItem
    extra = 0
    readonly_fields = ['product_name', 'price', 'quantity', 'subtotal']


@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ['order_no', 'total', 'item_count', 'created_at']
    list_display_links = ['order_no']
    search_fields = ['order_no']
    list_filter = ['created_at']
    list_per_page = 20
    ordering = ['-created_at']
    readonly_fields = ['order_no', 'total', 'item_count', 'created_at']
    inlines = [OrderItemInline]


@admin.register(OrderItem)
class OrderItemAdmin(admin.ModelAdmin):
    list_display = ['order', 'product_name', 'price', 'quantity', 'subtotal']
    list_filter = ['order']
    search_fields = ['product_name', 'order__order_no']
    list_per_page = 30
