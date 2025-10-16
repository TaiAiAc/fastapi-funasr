"""
服务层模块，包含各种业务逻辑的实现
为路由层提供统一的服务接口
"""

from .session_handler import SessionHandler

# 从vad模块导入VAD服务类和实例
from .vad import (
    VADService,
    vad_service,
    preload_vad_model,
    VADSession,
    StreamingVADService,
)

# 从kws模块导入KWS服务类和实例
from .kws import KWSService, get_kws_service, preload_kws_model

# 从asr模块导入ASR服务类和实例
from .asr import ASRService, get_asr_service, preload_asr_model

# 定义__all__列表，控制模块导出内容
__all__ = [
    "StreamingVADService",
    "VADService",
    "vad_service",
    "VADSession",
    "SessionHandler",
    "preload_vad_model",
    "KWSService",
    "get_kws_service",
    "preload_kws_model",
    "ASRService",
    "get_asr_service",
    "preload_asr_model",
]
