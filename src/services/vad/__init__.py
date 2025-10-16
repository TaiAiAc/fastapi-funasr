"""
VAD（语音端点检测）模块
提供语音活动检测、流式处理和会话管理功能
"""

from .core import VADService, vad_service, preload_vad_model
from .streaming import StreamingVADService


vad_service = VADService()


def preload_vad_model() -> bool:
    """预加载 VAD 模型"""
    try:
        vad_service.start()  # 调用基类的 start() 初始化模型
        return vad_service.is_initialized
    except Exception as e:
        error(f"预加载VAD模型失败: {e}")
        return False


__all__ = [
    "vad_service",
    "preload_vad_model",
    "StreamingVADService",
]
