"""
中间件管理器，用于统一注册所有中间件
"""
from ..utils.logger import logger
from .logging import LoggingMiddleware
from .authentication import AuthenticationMiddleware
from .error_handling import ErrorHandlingMiddleware
from .ip_whitelist import IPWhitelistMiddleware


class MiddlewareManager:
    """
    中间件管理器，用于统一注册所有中间件
    """
    @staticmethod
    def register_middlewares(app):
        """
        注册所有中间件到FastAPI应用
        
        参数:
            app: FastAPI应用实例
        """
        # 注册日志中间件
        app.add_middleware(LoggingMiddleware)
        
        # 注册认证中间件
        app.add_middleware(AuthenticationMiddleware)
        
        # 注册错误处理中间件
        app.add_middleware(ErrorHandlingMiddleware)
        
        # 注册IP白名单中间件
        # app.add_middleware(
        #     IPWhitelistMiddleware,
        #     whitelist=["127.0.0.1", "::1"],  # 可以在这里配置白名单
        #     exclude_paths=["/", "/docs", "/redoc", "/openapi.json"]  # 排除的路径
        # )
        app.add_middleware(
            IPWhitelistMiddleware,
            whitelist=None,  # None表示不限制任何IP
            exclude_paths=[]  # 空列表表示所有路径都不排除
        )
        
        # 可以根据需要添加更多中间件
        logger.info("所有中间件已成功注册")