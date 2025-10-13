"""
日志中间件，用于记录请求和响应信息
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
from ..utils import logger, log_request, log_response


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    日志中间件，记录请求和响应信息
    """
    async def dispatch(self, request: Request, call_next):
        # 记录请求开始时间
        start_time = time.time()
        
        # 记录请求信息 - 使用便捷函数
        log_request(request.method, request.url, request.client.host)
        
        # 处理请求
        response = await call_next(request)
        
        # 计算请求处理时间
        process_time = time.time() - start_time
        
        # 记录响应信息 - 使用便捷函数
        log_response(response.status_code, process_time)
        
        return response