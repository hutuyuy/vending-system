from django.shortcuts import redirect
from django.urls import reverse, resolve


class LoginRequiredMiddleware:
    """登录拦截中间件 - 未登录用户必须先登录才能访问主页面"""

    # 白名单：不需要登录即可访问的路径
    WHITELIST_PATHS = (
        '/login/',
        '/logout/',
        '/admin/login/',
    )

    # 白名单前缀：以这些开头的路径不需要登录
    WHITELIST_PREFIXES = (
        '/admin/',
        '/static/',
        '/media/',
    )

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        path = request.path

        # 已认证用户直接放行
        if request.user.is_authenticated:
            return self.get_response(request)

        # 检查精确白名单路径
        if path in self.WHITELIST_PATHS:
            return self.get_response(request)

        # 检查前缀白名单
        for prefix in self.WHITELIST_PREFIXES:
            if path.startswith(prefix):
                return self.get_response(request)

        # 未登录用户重定向到登录页
        return redirect('/login/')
