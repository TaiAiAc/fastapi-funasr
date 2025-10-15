# src/services/asr/streaming.py

from typing import Optional
import numpy as np
from ...utils import info, error

class StreamingASRService:
    """
    流式语音识别服务（基于 Paraformer online 模型）
    每个会话应创建独立实例
    """

    def __init__(self):
        info("【StreamingASR】初始化流式 ASR（暂未实现）")
        self.reset()

    def reset(self):
        """重置识别状态"""
        pass

    def feed_chunk(self, chunk: np.ndarray) -> Optional[str]:
        """
        输入音频块，返回部分识别结果（partial）或最终结果（final）
        """
        # TODO: 调用流式 Paraformer
        return None

    def finalize(self) -> Optional[str]:
        """强制结束识别，返回最终结果"""
        return None