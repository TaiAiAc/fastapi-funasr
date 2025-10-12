from fastapi import FastAPI
from .middleware import MiddlewareManager  # 导入方式保持不变
from .routes import register_routers

# 创建 FastAPI 应用实例
app = FastAPI(
    title="FunASR Web Service",
    description="A simple web service for speech recognition using FunASR",
    version="0.1.0",
)

# 注册中间件 - 使用方式保持不变
MiddlewareManager.register_middlewares(app)

# 注册路由
register_routers(app)


# 根路径路由
@app.get("/")
def read_root():
    return {"message": "Welcome to FunASR Web Service!"}


# 测试路由
@app.get("/hello")
def read_hello():
    return {"message": "Hello, FunASR!"}