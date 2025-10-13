from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .middleware import MiddlewareManager
from .routes import register_routers
from .services import preload_vad_model, vad_service
from .utils.logger import info, error

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
    info("应用启动中...")
    try:
        # 使用单独的线程预加载VAD模型
        import asyncio
        loop = asyncio.get_event_loop()
        
        # 预加载VAD模型
        vad_future = loop.run_in_executor(None, preload_vad_model)
        vad_success = await vad_future
        
        # 记录VAD模型加载状态
        if vad_success:
            info("VAD模型预加载成功！")
            # 记录VAD模型初始化信息
            vad_info = vad_service.get_init_info()
            info(f"VAD模型初始化信息: {vad_info}")
        else:
            error("VAD模型预加载失败，语音端点检测功能可能无法使用！")
            
    except Exception as e:
        error(f"启动过程中发生异常: {e}")


# 根路径路由 - 重定向到静态页面
from fastapi.responses import RedirectResponse

@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")