"""
IP白名单中间件，只允许白名单中的IP访问
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from ..utils.logger import logger


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """
    IP白名单中间件，只允许白名单中的IP访问
    """
    def __init__(self, app, whitelist=None, exclude_paths=None):
        """
        初始化白名单中间件
        
        参数:
            app: FastAPI应用实例
            whitelist: IP白名单列表，为None时允许所有IP访问
            exclude_paths: 不进行白名单检查的路径列表
        """
        super().__init__(app)
        # 设置whitelist为None时允许所有IP访问
        self.whitelist = whitelist  # 不再设置默认值
        # 默认排除路径，如文档页面和根路径
        self.exclude_paths = exclude_paths or ["/", "/docs", "/redoc", "/openapi.json"]
        
        if whitelist is None:
            logger.info("IP白名单已禁用，允许所有IP访问")
        else:
            logger.info(f"IP白名单已初始化: {self.whitelist}")
        logger.info(f"排除白名单检查的路径: {self.exclude_paths}")
    
    async def dispatch(self, request: Request, call_next):
        # 获取客户端IP
        client_ip = request.client.host
        
        # 检查路径是否在排除列表中
        if request.url.path in self.exclude_paths:
            # 路径在排除列表中，直接处理请求
            return await call_next(request)
        
        # 检查是否禁用了IP白名单
        if self.whitelist is None:
            # IP白名单已禁用，允许所有IP访问
            return await call_next(request)
        
        # 检查IP是否在白名单中
        if client_ip not in self.whitelist:
            # IP不在白名单中，记录日志并拒绝访问
            logger.warning(f"拒绝非白名单IP访问: {client_ip} 请求路径: {request.url.path}")
            return Response(
                status_code=403,
                content="Forbidden: IP not in whitelist"
            )
        
        # IP在白名单中，处理请求
        return await call_next(request)