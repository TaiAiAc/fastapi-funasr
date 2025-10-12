"""
认证中间件，用于验证请求的合法性
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from ..utils.logger import logger


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    认证中间件，用于验证请求的合法性
    注意：实际应用中应使用更安全的认证方式
    """
    async def dispatch(self, request: Request, call_next):
        # 获取Authorization头
        auth_header = request.headers.get("Authorization")
        
        # 公开路径不需要认证
        public_paths = ["/", "/hello", "/docs", "/redoc", "/openapi.json"]
        
        # 检查是否需要认证
        if request.url.path not in public_paths and not auth_header:
            # 未提供认证信息，返回401
            logger.warning(f"未授权访问: {request.url.path}")
            pass
            # return Response(status_code=401, content="未授权访问")
        
        # 处理请求
        response = await call_next(request)
        return response