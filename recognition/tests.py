from django.test import TestCase, Client
from django.contrib.auth.models import User
from .models import Product, Order, OrderItem


class LoginViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='testpass123')

    def test_login_page_loads(self):
        response = self.client.get('/login/')
        self.assertEqual(response.status_code, 200)

    def test_login_success(self):
        response = self.client.post('/login/', {'username': 'testuser', 'password': 'testpass123'})
        self.assertEqual(response.status_code, 302)

    def test_login_failure(self):
        response = self.client.post('/login/', {'username': 'testuser', 'password': 'wrong'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '用户名或密码错误')


class ProductModelTest(TestCase):
    def test_create_product(self):
        product = Product.objects.create(name='测试商品', price=9.99, stock=100)
        self.assertEqual(str(product), '测试商品 - ¥9.99')

    def test_low_stock_property(self):
        product = Product(name='测试', price=1.00, stock=3, low_stock_threshold=5)
        self.assertTrue(product.is_low_stock)


class OrderModelTest(TestCase):
    def test_create_order(self):
        order = Order.objects.create(order_no='TEST001', total=19.98, item_count=2)
        self.assertEqual(str(order), 'TEST001 - ¥19.98')
