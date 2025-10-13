"""
错误处理中间件，捕获并处理请求过程中的异常
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from ..utils import logger


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    错误处理中间件，捕获并处理请求过程中的异常
    """
    async def dispatch(self, request: Request, call_next):
        try:
            # 尝试处理请求
            response = await call_next(request)
            return response
        except Exception as e:
            # 捕获异常并记录
            logger.error(f"请求处理异常: {str(e)}")
            # 返回统一的错误响应
            return Response(status_code=500, content=f"服务器内部错误: {str(e)}")