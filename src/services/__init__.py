# services模块初始化文件
# 用于存放各种业务服务实现

"""
服务层模块，包含各种业务逻辑的实现
为路由层提供统一的服务接口
"""

# 从vad模块导入VAD服务类和实例
from .vad import (
    VADService,
    vad_service,
    preload_vad_model
)

# 定义__all__列表，控制模块导出内容
__all__ = [
    "VADService",
    "vad_service",
    "preload_vad_model"
]