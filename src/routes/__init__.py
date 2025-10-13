from fastapi import FastAPI
from ..utils import logger

# 导入各个路由模块
from .recognition import recognition_router
from .funasr import websocket_router  # 更新导入名称


def register_routers(app: FastAPI):
    """
    注册所有路由器到主应用
    """
    # 注册各个路由模块
    app.include_router(recognition_router)
    app.include_router(websocket_router)  # 使用新名称
    # 可以继续添加其他路由模块
    
    logger.info("所有路由已成功注册")