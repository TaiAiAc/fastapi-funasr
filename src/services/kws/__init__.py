from .core import KWSService

# 创建全局KWS服务实例
kws_service = KWSService()

def preload_kws_model() -> bool:
    """预加载 KWS 模型"""
    try:
        kws_service.start()  # 调用基类的 start() 初始化模型
        return kws_service.is_initialized
    except Exception as e:
        error(f"预加载KWS模型失败: {e}")
        return False
