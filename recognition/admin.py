import csv
import datetime

from django.contrib import admin
from django.db.models import Count, F, Sum, Q
from django.db.models.functions import TruncDate
from django.http import HttpResponse, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import path
from django.utils import timezone
from django.utils.html import format_html
from django.utils.safestring import mark_safe

from .models import Product, ProductImage, Order, OrderItem, RestockRecord


# ============================================================
#  Inline
# ============================================================

class ProductImageInline(admin.TabularInline):
    model = ProductImage
    extra = 1
    fields = ['image', 'view_angle', 'image_preview']
    readonly_fields = ['image_preview']

    def image_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{0}" style="max-height:80px; border-radius:6px;" />',
                obj.image.url
            )
        return '-'
    image_preview.short_description = '预览'


class OrderItemInline(admin.TabularInline):
    model = OrderItem
    extra = 0
    readonly_fields = ['product_name', 'price', 'quantity', 'subtotal']


# ============================================================
#  自定义日期筛选器
# ============================================================

class OrderDateFilter(admin.SimpleListFilter):
    title = '下单时间'
    parameter_name = 'order_date'

    def lookups(self, request, model_admin):
        return (
            ('today', '今天'),
            ('yesterday', '昨天'),
            ('this_week', '本周'),
            ('this_month', '本月'),
        )

    def queryset(self, request, queryset):
        now = timezone.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if self.value() == 'today':
            return queryset.filter(created_at__gte=today)
        if self.value() == 'yesterday':
            yesterday = today - datetime.timedelta(days=1)
            return queryset.filter(created_at__gte=yesterday, created_at__lt=today)
        if self.value() == 'this_week':
            week_start = today - datetime.timedelta(days=today.weekday())
            return queryset.filter(created_at__gte=week_start)
        if self.value() == 'this_month':
            month_start = today.replace(day=1)
            return queryset.filter(created_at__gte=month_start)
        return queryset


# ============================================================
#  ProductAdmin
# ============================================================

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'price', 'stock_display', 'image_count',
        'is_active_badge', 'status_badge', 'created_at',
    ]
    list_display_links = ['name']
    search_fields = ['name', 'description']
    date_hierarchy = 'created_at'
    list_per_page = 20
    ordering = ['-created_at']
    inlines = [ProductImageInline]
    list_filter = ['is_active']
    actions = ['activate_products', 'deactivate_products']

    fieldsets = (
        ('基本信息', {
            'fields': ('name', 'price', 'description', 'is_active'),
        }),
        ('库存管理', {
            'fields': ('stock', 'low_stock_threshold'),
        }),
    )

    # ---------- queryset 优化 ----------

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.annotate(_image_count=Count('images'))

    # ---------- 显示字段 ----------

    def stock_display(self, obj):
        """库存数量，低库存红色高亮"""
        if hasattr(obj, 'stock'):
            threshold = getattr(obj, 'low_stock_threshold', 5)
            if obj.stock == 0:
                return format_html(
                    '<span style="color:#ef4444; font-weight:700;">{} 🔴</span>', obj.stock
                )
            elif obj.stock <= threshold:
                return format_html(
                    '<span style="color:#f59e0b; font-weight:600;">{} ⚠️</span>', obj.stock
                )
            return format_html(
                '<span style="color:#10b981; font-weight:600;">{}</span>', obj.stock
            )
        return '-'
    stock_display.short_description = '库存'

    def image_count(self, obj):
        count = obj._image_count  # annotate 预取
        if count >= 8:
            return format_html('<span style="color:#10b981; font-weight:600;">{} 张 ✅</span>', count)
        elif count > 0:
            return format_html('<span style="color:#f59e0b; font-weight:600;">{} 张 ⚠️</span>', count)
        return format_html('<span style="color:#ef4444;">{} 张 ❌</span>', count)
    image_count.short_description = '图片数量'

    def is_active_badge(self, obj):
        if obj.is_active:
            return mark_safe(
                '<span style="background:rgba(16,185,129,0.15); color:#059669; '
                'padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600;">已上架</span>'
            )
        return mark_safe(
            '<span style="background:rgba(239,68,68,0.15); color:#dc2626; '
            'padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600;">已下架</span>'
        )
    is_active_badge.short_description = '上架状态'

    def status_badge(self, obj):
        count = obj._image_count
        if count >= 8:
            return mark_safe(
                '<span style="background:rgba(16,185,129,0.15); color:#059669; padding:3px 10px; '
                'border-radius:12px; font-size:12px; font-weight:600;">就绪</span>'
            )
        elif count > 0:
            return mark_safe(
                '<span style="background:rgba(245,158,11,0.15); color:#d97706; padding:3px 10px; '
                'border-radius:12px; font-size:12px; font-weight:600;">图片不足</span>'
            )
        return mark_safe(
            '<span style="background:rgba(239,68,68,0.15); color:#dc2626; padding:3px 10px; '
            'border-radius:12px; font-size:12px; font-weight:600;">无图片</span>'
        )
    status_badge.short_description = '图片状态'

    # ---------- 商品详情 change_view ----------

    def change_view(self, request, object_id, form_url='', extra_context=None):
        extra_context = extra_context or {}
        product = self.get_object(request, object_id)
        if product:
            extra_context['image_list'] = product.images.all().order_by('view_angle')
            if hasattr(product, 'restock_records'):
                extra_context['restock_records'] = product.restock_records.all()[:10]
        return super().change_view(request, object_id, form_url, extra_context)

    # ---------- 批量上下架 actions ----------

    @admin.action(description='✅ 批量上架选中商品')
    def activate_products(self, request, queryset):
        count = queryset.update(is_active=True)
        self.message_user(request, f'成功上架 {count} 个商品。')

    @admin.action(description='❌ 批量下架选中商品')
    def deactivate_products(self, request, queryset):
        count = queryset.update(is_active=False)
        self.message_user(request, f'成功下架 {count} 个商品。')


# ============================================================
#  ProductImageAdmin
# ============================================================

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
                '<img src="{0}" style="max-height:60px; border-radius:4px;" />',
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


# ============================================================
#  RestockRecordAdmin
# ============================================================

@admin.register(RestockRecord)
class RestockRecordAdmin(admin.ModelAdmin):
    list_display = ['product', 'quantity', 'operator', 'note', 'created_at']
    list_display_links = ['product']
    search_fields = ['product__name', 'operator', 'note']
    list_filter = ['created_at']
    list_per_page = 20
    ordering = ['-created_at']
    readonly_fields = ['created_at']


# ============================================================
#  OrderAdmin
# ============================================================

@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ['order_no', 'total', 'item_count', 'created_at']
    list_display_links = ['order_no']
    search_fields = ['order_no']
    list_filter = [OrderDateFilter]
    list_per_page = 20
    ordering = ['-created_at']
    readonly_fields = ['order_no', 'total', 'item_count', 'created_at']
    inlines = [OrderItemInline]
    actions = ['export_orders']

    # ---------- 聚合优化 ----------

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.annotate(_total_amount=Sum('items__subtotal'))

    def total_amount(self, obj):
        return obj._total_amount or obj.total
    total_amount.short_description = '订单金额(聚合)'

    # ---------- 导出 action ----------

    @admin.action(description='📥 导出选中订单为 CSV')
    def export_orders(self, request, queryset):
        response = HttpResponse(content_type='text/csv; charset=utf-8-sig')
        response['Content-Disposition'] = 'attachment; filename="orders_export.csv"'

        # BOM for Excel
        response.write('\ufeff')

        writer = csv.writer(response)
        writer.writerow(['订单号', '总计', '商品种类数', '下单时间'])

        for order in queryset:
            writer.writerow([
                order.order_no,
                order.total,
                order.item_count,
                order.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            ])

        return response


# ============================================================
#  OrderItemAdmin
# ============================================================

@admin.register(OrderItem)
class OrderItemAdmin(admin.ModelAdmin):
    list_display = ['order', 'product_name', 'price', 'quantity', 'subtotal']
    list_filter = ['order']
    search_fields = ['product_name', 'order__order_no']
    list_per_page = 30


# ============================================================
#  Dashboard 自定义首页
# ============================================================

# 保存原始 get_urls
_original_get_urls = admin.site.get_urls

def _custom_get_urls():
    """为默认 admin.site 注入 dashboard 路由"""
    custom_urls = [
        path('dashboard/', admin.site.admin_view(dashboard_view), name='dashboard'),
    ]
    return custom_urls + _original_get_urls()

def dashboard_view(request):
    now = timezone.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    today_orders = Order.objects.filter(created_at__gte=today_start)
    today_order_count = today_orders.count()
    today_revenue = today_orders.aggregate(total=Sum('total'))['total'] or 0

    product_total = Product.objects.count()
    low_stock_count = Product.objects.filter(
        stock__lte=F('low_stock_threshold')
    ).count()

    seven_days_ago = today_start - datetime.timedelta(days=6)
    daily_orders = (
        Order.objects
        .filter(created_at__gte=seven_days_ago)
        .annotate(date=TruncDate('created_at'))
        .values('date')
        .annotate(count=Count('id'), revenue=Sum('total'))
        .order_by('date')
    )
    trend = []
    date_map = {item['date']: item for item in daily_orders}
    for i in range(7):
        d = (seven_days_ago + datetime.timedelta(days=i)).date()
        entry = date_map.get(d, {'count': 0, 'revenue': 0})
        trend.append({
            'date': d.strftime('%m-%d'),
            'count': entry['count'],
            'revenue': float(entry['revenue'] or 0),
        })

    context = {
        **admin.site.each_context(request),
        'title': '系统概览',
        'today_order_count': today_order_count,
        'today_revenue': today_revenue,
        'product_total': product_total,
        'low_stock_count': low_stock_count,
        'trend': trend,
    }
    return TemplateResponse(request, 'admin/dashboard.html', context)

# 注入自定义路由
admin.site.get_urls = _custom_get_urls

# 设置站点标题
admin.site.site_header = '🛒 智能售货系统 - 管理后台'
admin.site.site_title = '智能售货系统'
admin.site.index_title = '系统管理'
