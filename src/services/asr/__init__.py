from .core import ASRService
from .streaming import StreamingASRService

asr_service = ASRService()

def preload_asr_model() -> bool:
    """预加载 ASR 模型"""
    try:
        asr_service.start()  # 调用基类的 start() 初始化模型
        return asr_service.is_initialized
    except Exception as e:
        error(f"预加载ASR模型失败: {e}")
        return False


__all__ = ["preload_asr_model", "StreamingASRService"]
