import json
import os

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .models import Product
from .ml.predictor import predict_image, get_predictor


def index(request):
    products = Product.objects.all()
    return render(request, 'recognition/index.html', {'products': products})


def product_list(request):
    products = Product.objects.all()
    return render(request, 'recognition/product_list.html', {'products': products})


def checkout(request):
    return render(request, 'recognition/checkout.html')


def evaluation(request):
    return render(request, 'recognition/evaluation.html')


@csrf_exempt
def recognize_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded = request.FILES['image']
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, 'capture.jpg')
        with open(temp_path, 'wb+') as f:
            for chunk in uploaded.chunks():
                f.write(chunk)
        result = predict_image(temp_path)
        if result['success']:
            try:
                product = Product.objects.get(name=result['product_name'])
                result['price'] = str(product.price)
            except Product.DoesNotExist:
                result['price'] = '0.00'
        return JsonResponse(result)
    return JsonResponse({'success': False, 'message': '请上传图片'})


@csrf_exempt
def checkout_submit(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        items = data.get('items', [])
        total = sum(float(i['price']) * i['qty'] for i in items)
        return JsonResponse({
            'success': True,
            'total': round(total, 2),
            'message': f'结算成功！总计：¥{round(total, 2)}',
        })
    return JsonResponse({'success': False, 'message': '错误'})


@csrf_exempt
def train_model_view(request):
    if request.method == 'POST':
        from .ml.trainer import train_model
        result = train_model(epochs=30, batch_size=8, learning_rate=0.001, patience=5)
        return JsonResponse(result)
    return JsonResponse({'success': False, 'message': '请使用 POST 请求'})


@csrf_exempt
def training_history(request):
    from .ml.trainer import load_history
    history = load_history()
    if history:
        return JsonResponse({'success': True, 'history': history})
    return JsonResponse({'success': False, 'message': '暂无训练历史，请先训练模型'})
