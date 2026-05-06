from django.urls import path
from . import views

app_name = 'recognition'

urlpatterns = [
    path('', views.index, name='index'),
    path('products/', views.product_list, name='product_list'),
    path('checkout/', views.checkout, name='checkout'),
    path('orders/', views.order_history, name='order_history'),
    path('evaluation/', views.evaluation, name='evaluation'),
    path('api/recognize/', views.recognize_image, name='recognize'),
    path('api/checkout/', views.checkout_submit, name='checkout_submit'),
    path('api/train/', views.train_model_view, name='train'),
    path('api/train/status/', views.training_status, name='train_status'),
    path('api/history/', views.training_history, name='training_history'),
]
