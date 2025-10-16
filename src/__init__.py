# 尝试加载.env文件中的环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()  # 加载.env文件中的环境变量
except ImportError:
    error("dotenv模块未安装，无法加载.env文件中的环境变量")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .middleware import MiddlewareManager
from .routes import register_routers
from .services import preload_vad_model
from .services import preload_kws_model
from .services import preload_asr_model
from .utils import info,debug, error
import os

# 创建 FastAPI 应用实例
app = FastAPI(
    title="Web Service",
    description="A simple web service",
    version="0.1.0",
)

# 挂载静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

# 注册中间件
MiddlewareManager.register_middlewares(app)

# 注册路由
register_routers(app)


# 应用启动时预加载模型
@app.on_event("startup")
async def startup_event():
    """应用启动事件，用于预加载VAD模型"""

    # 尝试加载.env文件中的环境变量
    test_env = os.getenv("APP_TEST")
    debug(test_env)

    # 导入FunASR相关库
    try:
        from funasr import __version__
        import torch

        info(f"FunASR版本: {__version__}")
    except ImportError as e:
        error(f"导入FunASR相关库失败: {e}")
        raise

    try:
        # 使用单独的线程预加载模型
        import asyncio

        loop = asyncio.get_event_loop()

        loop.run_in_executor(None, preload_vad_model)
        loop.run_in_executor(None, preload_kws_model)
        loop.run_in_executor(None, preload_asr_model)
    except Exception as e:
        error(f"启动过程中发生异常: {e}")


# 根路径路由 - 重定向到静态页面
from fastapi.responses import RedirectResponse


@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")
