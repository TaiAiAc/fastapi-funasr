# src/services/kws/streaming.py

from typing import Optional
import numpy as np
from ...utils import info, error
from .core import get_kws_service

class StreamingKWSService:
    """
    流式关键词检测服务（专为 xiaoyun 唤醒词设计）
    """

    def __init__(self):
        self.kws_service = get_kws_service()
        if not self.kws_service.is_initialized():
            raise RuntimeError("KWS 模型未初始化")
        self.reset()

    def reset(self):
        self.cache = None
        self.is_keyword_detected = False

    def detect_keyword_stream(self, chunk: np.ndarray) -> bool:
        """
        输入音频块，返回是否检测到唤醒词
        """
        # TODO: 实现流式推理（后续填充）
        return False

    def is_detected(self) -> bool:
        return self.is_keyword_detected