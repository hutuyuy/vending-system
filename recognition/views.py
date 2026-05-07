import json
import os
import uuid
import logging
import threading
import time

from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST

from .models import Product, Order, OrderItem, RestockRecord

logger = logging.getLogger('recognition')


def login_view(request):
    """登录页面"""
    if request.user.is_authenticated:
        return redirect('recognition:index')

    error = None
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('recognition:index')
        else:
            error = '用户名或密码错误，请重试'

    return render(request, 'recognition/login.html', {'error': error})


def logout_view(request):
    """登出"""
    logout(request)
    return redirect('recognition:login')


# 允许的图片类型和最大大小
ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/png', 'image/webp'}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

# 训练锁，防止并发训练
_training_lock = threading.Lock()
_training_status = {'running': False, 'progress': '', 'result': None, 'started_at': 0}


def _validate_image(file):
    """校验上传的图片文件"""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        return False, f'不支持的文件类型: {file.content_type}，仅支持 JPEG/PNG/WebP'
    if file.size > MAX_IMAGE_SIZE:
        return False, f'文件过大: {file.size / 1024 / 1024:.1f}MB，最大允许 10MB'
    return True, ''


def index(request):
    from django.db.models import Sum, Count
    products = Product.objects.prefetch_related('images').all()
    recent_orders = Order.objects.prefetch_related('items').all()[:5]
    order_stats = Order.objects.aggregate(count=Count('id'), revenue=Sum('total'))
    stats = {
        'product_count': products.count(),
        'image_count': sum(p.images.count() for p in products),
        'order_count': order_stats['count'] or 0,
        'total_revenue': float(order_stats['revenue'] or 0),
    }
    return render(request, 'recognition/index.html', {
        'products': products,
        'recent_orders': recent_orders,
        'stats': stats,
    })


def product_list(request):
    products = Product.objects.prefetch_related('images').all()
    return render(request, 'recognition/product_list.html', {'products': products})


def checkout(request):
    return render(request, 'recognition/checkout.html')


def evaluation(request):
    return render(request, 'recognition/evaluation.html')


def order_history(request):
    orders = Order.objects.prefetch_related('items').all()[:50]
    return render(request, 'recognition/order_history.html', {'orders': orders})


@require_POST
def recognize_image(request):
    """图片识别接口"""
    if not request.FILES.get('image'):
        return JsonResponse({'success': False, 'message': '请上传图片'}, status=400)

    uploaded = request.FILES['image']

    # 校验文件
    valid, msg = _validate_image(uploaded)
    if not valid:
        logger.warning(f'图片校验失败: {msg}')
        return JsonResponse({'success': False, 'message': msg}, status=400)

    # 用唯一文件名保存，避免并发冲突
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f'{uuid.uuid4().hex}.jpg')

    try:
        with open(temp_path, 'wb+') as f:
            for chunk in uploaded.chunks():
                f.write(chunk)

        logger.info(f'图片已保存: {temp_path} ({uploaded.size} bytes)')
        from .ml.predictor import predict_image
        result = predict_image(temp_path)

        if result['success']:
            try:
                product = Product.objects.get(name=result['product_name'])
                result['price'] = str(product.price)
            except Product.DoesNotExist:
                result['price'] = '0.00'
                logger.warning(f'识别到未知商品: {result["product_name"]}')

        return JsonResponse(result)
    except Exception as e:
        logger.exception(f'识别过程出错: {e}')
        return JsonResponse({'success': False, 'message': f'识别失败: {str(e)}'}, status=500)
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


@require_POST
def checkout_submit(request):
    """结算提交接口"""
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'success': False, 'message': '请求数据格式错误'}, status=400)

    items = data.get('items', [])
    if not items:
        return JsonResponse({'success': False, 'message': '购物车为空'}, status=400)

    # 校验每个商品
    total = 0
    validated_items = []
    for item in items:
        name = item.get('name', '')
        price = item.get('price', 0)
        qty = item.get('qty', 1)

        if not name or not isinstance(qty, (int, float)) or qty < 1:
            continue

        try:
            price = float(price)
        except (ValueError, TypeError):
            continue

        subtotal = price * int(qty)
        total += subtotal
        validated_items.append({
            'name': name,
            'price': price,
            'qty': int(qty),
            'subtotal': round(subtotal, 2),
        })

    if not validated_items:
        return JsonResponse({'success': False, 'message': '没有有效的商品'}, status=400)

    # 检查库存
    for item in validated_items:
        try:
            product = Product.objects.get(name=item['name'])
            if product.stock < item['qty']:
                return JsonResponse({
                    'success': False,
                    'message': f'商品「{item["name"]}」库存不足，当前库存: {product.stock}',
                }, status=400)
            if product.stock == 0:
                return JsonResponse({
                    'success': False,
                    'message': f'商品「{item["name"]}」已缺货',
                }, status=400)
        except Product.DoesNotExist:
            pass

    total = round(total, 2)

    # 扣减库存
    for item in validated_items:
        try:
            product = Product.objects.get(name=item['name'])
            product.stock = max(0, product.stock - item['qty'])
            product.save()
        except Product.DoesNotExist:
            pass

    # 保存订单到数据库
    import datetime
    order_no = datetime.datetime.now().strftime('%Y%m%d%H%M%S') + uuid.uuid4().hex[:6].upper()
    order = Order.objects.create(
        order_no=order_no,
        total=total,
        item_count=len(validated_items),
    )
    for item in validated_items:
        OrderItem.objects.create(
            order=order,
            product_name=item['name'],
            price=item['price'],
            quantity=item['qty'],
            subtotal=item['subtotal'],
        )

    logger.info(f'结算完成: 订单{order_no}, {len(validated_items)} 种商品, 总计 ¥{total}')

    return JsonResponse({
        'success': True,
        'total': total,
        'items': validated_items,
        'order_no': order_no,
        'message': f'结算成功！订单号：{order_no}，总计：¥{total}',
    })


@require_POST
def train_model_view(request):
    """训练模型接口 - 使用线程异步执行（需要登录）"""
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'message': '请先登录后再操作'}, status=403)

    if _training_status['running']:
        return JsonResponse({
            'success': False,
            'message': '模型正在训练中，请稍后再试',
        })

    def _run_training():
        _training_status['running'] = True
        _training_status['progress'] = '训练中...'
        _training_status['started_at'] = time.time()
        try:
            from .ml.trainer import train_model
            result = train_model(epochs=30, batch_size=8, learning_rate=0.001, patience=5)
            _training_status['result'] = result
            _training_status['progress'] = '完成' if result.get('success') else '失败'
            logger.info(f'训练完成: {result}')
        except Exception as e:
            _training_status['result'] = {'success': False, 'message': f'训练异常: {str(e)}'}
            _training_status['progress'] = '异常'
            logger.exception(f'训练异常: {e}')
        finally:
            _training_status['running'] = False

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()

    logger.info('训练任务已启动')
    return JsonResponse({
        'success': True,
        'message': '训练已启动，请稍候查看结果',
    })


def training_status(request):
    """查询训练状态"""
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'message': '请先登录'}, status=403)

    # 超时保护：训练超过30分钟视为异常
    if _training_status['running'] and time.time() - _training_status.get('started_at', 0) > 1800:
        _training_status['running'] = False
        _training_status['progress'] = '超时'
        _training_status['result'] = {'success': False, 'message': '训练超时（超过30分钟），请重试'}

    return JsonResponse({
        'running': _training_status['running'],
        'progress': _training_status['progress'],
        'result': _training_status['result'],
    })


def training_history(request):
    """获取训练历史"""
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'message': '请先登录'}, status=403)
    from .ml.trainer import load_history
    history = load_history()
    if history:
        return JsonResponse({'success': True, 'history': history})
    return JsonResponse({'success': False, 'message': '暂无训练历史，请先训练模型'})


def restock_page(request):
    """补货管理页面"""
    products = Product.objects.prefetch_related('images').all()
    return render(request, 'recognition/restock.html', {'products': products})


@require_POST
def restock_api(request):
    """补货 API"""
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'success': False, 'message': '请求数据格式错误'}, status=400)

    product_id = data.get('product_id')
    quantity = data.get('quantity')

    if not product_id or not quantity:
        return JsonResponse({'success': False, 'message': '缺少参数'}, status=400)

    try:
        quantity = int(quantity)
        if quantity <= 0:
            raise ValueError
    except (ValueError, TypeError):
        return JsonResponse({'success': False, 'message': '补货数量必须为正整数'}, status=400)

    try:
        product = Product.objects.get(pk=product_id)
    except Product.DoesNotExist:
        return JsonResponse({'success': False, 'message': '商品不存在'}, status=404)

    product.stock += quantity
    product.save()

    note = data.get('note', '')
    operator = data.get('operator', '')
    RestockRecord.objects.create(
        product=product, quantity=quantity,
        note=note, operator=operator,
    )

    logger.info(f'补货: {product.name} +{quantity}')

    return JsonResponse({
        'success': True,
        'message': f'{product.name} 补货 {quantity} 件成功',
        'new_stock': product.stock,
    })
