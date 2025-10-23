# src/services/kws/streaming.py

from ...utils import debug
import numpy as np
from typing import Optional, Dict, Any

class StreamingKWSService:
    def __init__(self, kws_service):
        """
        使用外部 KWSService 的模型，避免重复加载
        """
        self.kws_service = kws_service
        self.reset()

    def reset(self):
        """重置流式状态"""
        self.cache = {}
        self.is_active = True  # 可用于状态机控制

   