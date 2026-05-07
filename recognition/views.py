"""
views.py — 补丁说明：
1. 新增 barcode_scan：条形码/二维码识别接口
2. recognize_image 改为支持多图上传 + TTA
3. 新增 recognize_multi：多图投票识别接口
"""
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


def _save_temp(file):
    """保存上传文件到临时目录，返回路径"""
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f'{uuid.uuid4().hex}.jpg')
    with open(temp_path, 'wb+') as f:
        for chunk in file.chunks():
            f.write(chunk)
    return temp_path


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


# ============================================================
#  条形码 / 二维码识别（新增）
# ============================================================

@require_POST
def barcode_scan(request):
    """条形码/二维码扫描接口"""
    if not request.FILES.get('image'):
        return JsonResponse({'success': False, 'message': '请上传图片'}, status=400)

    uploaded = request.FILES['image']
    valid, msg = _validate_image(uploaded)
    if not valid:
        return JsonResponse({'success': False, 'message': msg}, status=400)

    temp_path = None
    try:
        temp_path = _save_temp(uploaded)

        # 用 pyzbar 解码
        try:
            from pyzbar.pyzbar import decode as pyzbar_decode
        except ImportError:
            return JsonResponse({
                'success': False,
                'message': 'pyzbar 未安装，请运行: pip install pyzbar',
            }, status=500)

        from PIL import Image
        img = Image.open(temp_path)
        decoded = pyzbar_decode(img)

        if not decoded:
            return JsonResponse({
                'success': False,
                'message': '未识别到条形码或二维码，请对准商品条码重试',
            })

        # 取第一个识别到的码
        code = decoded[0]
        code_data = code.data.decode('utf-8', errors='replace')
        code_type = code.type  # 'QRCODE', 'EAN13', 'CODE128', etc.

        logger.info(f'扫码成功: {code_data} (类型: {code_type})')

        # 尝试用条码匹配商品（需要在 Product 模型里加 barcode 字段，
        # 或者用商品名称匹配条码内容）
        product = None

        # 方案1：按 barcode 字段精确匹配
        try:
            product = Product.objects.get(barcode=code_data)
        except (Product.DoesNotExist, Exception):
            pass

        # 方案2：条码内容可能就是商品名称
        if not product:
            try:
                product = Product.objects.get(name__icontains=code_data)
            except Product.DoesNotExist:
                pass

        # 方案3：条码内容包含商品名
        if not product:
            for p in Product.objects.filter(is_active=True):
                if p.name in code_data or code_data in p.name:
                    product = p
                    break

        if product:
            return JsonResponse({
                'success': True,
                'product_name': product.name,
                'price': str(product.price),
                'stock': product.stock,
                'code': code_data,
                'code_type': code_type,
                'match_method': 'barcode',
            })
        else:
            # 未匹配到 → 返回商品列表供用户选择绑定
            all_products = list(Product.objects.filter(is_active=True).values('id', 'name', 'price'))
            return JsonResponse({
                'success': False,
                'message': f'扫码成功（{code_data}），但未匹配到商品。请选择商品绑定。',
                'code': code_data,
                'code_type': code_type,
                'need_bind': True,
                'products': all_products,
            })

    except Exception as e:
        logger.exception(f'扫码失败: {e}')
        return JsonResponse({'success': False, 'message': f'扫码失败: {str(e)}'}, status=500)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ============================================================
#  条码绑定（新增）
# ============================================================

@require_POST
def barcode_bind(request):
    """将扫码结果绑定到指定商品"""
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'success': False, 'message': '请求数据格式错误'}, status=400)

    code = data.get('code', '').strip()
    product_id = data.get('product_id')

    if not code or not product_id:
        return JsonResponse({'success': False, 'message': '缺少条码或商品ID'}, status=400)

    try:
        product = Product.objects.get(pk=product_id)
    except Product.DoesNotExist:
        return JsonResponse({'success': False, 'message': '商品不存在'}, status=404)

    # 保存条码到商品
    product.barcode = code
    product.save()

    logger.info(f'条码绑定: {product.name} ← {code}')

    return JsonResponse({
        'success': True,
        'message': f'已将条码 {code} 绑定到「{product.name}」',
        'product_name': product.name,
        'price': str(product.price),
        'stock': product.stock,
    })


def barcode_lookup(request):
    """手动输入条码查询（GET）"""
    code = request.GET.get('code', '').strip()
    if not code:
        return JsonResponse({'success': False, 'message': '请输入条码'}, status=400)

    product = None
    try:
        product = Product.objects.get(barcode=code)
    except (Product.DoesNotExist, Exception):
        pass

    if not product:
        try:
            product = Product.objects.get(name__icontains=code)
        except Product.DoesNotExist:
            pass

    if product:
        return JsonResponse({
            'success': True,
            'product_name': product.name,
            'price': str(product.price),
            'stock': product.stock,
            'code': code,
        })
    else:
        all_products = list(Product.objects.filter(is_active=True).values('id', 'name', 'price'))
        return JsonResponse({
            'success': False,
            'message': f'未找到条码 {code} 对应的商品，请选择绑定。',
            'code': code,
            'need_bind': True,
            'products': all_products,
        })


# ============================================================
#  图片识别（改进：支持多图 + TTA）
# ============================================================

@require_POST
def recognize_image(request):
    """图片识别接口 — 支持多文件上传"""
    files = request.FILES.getlist('image')
    if not files:
        return JsonResponse({'success': False, 'message': '请上传图片'}, status=400)

    # 校验
    temp_paths = []
    try:
        for uploaded in files:
            valid, msg = _validate_image(uploaded)
            if not valid:
                return JsonResponse({'success': False, 'message': msg}, status=400)
            temp_paths.append(_save_temp(uploaded))

        logger.info(f'识别请求: {len(temp_paths)} 张图片')

        from .ml.predictor import get_predictor
        predictor = get_predictor()

        if len(temp_paths) == 1:
            result = predictor.predict(temp_paths[0])
        else:
            result = predictor.predict_multiple(temp_paths)

        if result['success']:
            try:
                product = Product.objects.get(name=result['product_name'])
                result['price'] = str(product.price)
                result['stock'] = product.stock
            except Product.DoesNotExist:
                result['price'] = '0.00'
                logger.warning(f'识别到未知商品: {result["product_name"]}')

        return JsonResponse(result)

    except Exception as e:
        logger.exception(f'识别过程出错: {e}')
        return JsonResponse({'success': False, 'message': f'识别失败: {str(e)}'}, status=500)
    finally:
        for p in temp_paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


# ============================================================
#  结算
# ============================================================

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


# ============================================================
#  训练相关
# ============================================================

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


# ============================================================
#  补货
# ============================================================

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
