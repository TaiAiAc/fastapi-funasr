# src/services/kws/streaming.py

from typing import Optional
import numpy as np
from ...utils import info, error, debug


class StreamingKWSService:
    """
    流式语音端点检测服务（专为 xiaoyun 唤醒词设计）
    通过维护滑动窗口缓冲区模拟流式检测
    """

    def __init__(
        self, model, buffer_duration_sec: float = 1.5, sample_rate: int = 16000
    ):
        """
        Args:
            buffer_duration_sec: 缓冲区最大时长（秒），建议 1.0 ~ 1.8
            sample_rate: 音频采样率，固定为 16000
        """
        self.model = model
        if not self.model.is_initialized:
            raise RuntimeError("KWS 模型未初始化")
        self.sample_rate = sample_rate
        self.max_buffer_samples = int(buffer_duration_sec * sample_rate)
        self.reset()

    def reset(self):
        """重置内部状态"""
        self._buffer: list[np.ndarray] = []
        self._total_samples = 0
        self.is_keyword_detected = False

    def _trim_buffer(self):
        """确保缓冲区不超过最大长度"""
        while self._total_samples > self.max_buffer_samples and self._buffer:
            removed = self._buffer.pop(0)
            self._total_samples -= len(removed)

    def _get_buffered_audio(self) -> Optional[np.ndarray]:
        """获取当前缓冲区的完整音频"""
        if not self._buffer:
            return None
        if len(self._buffer) == 1:
            return self._buffer[0].copy()
        return np.concatenate(self._buffer)

    def detect_keyword_stream(self, chunk: np.ndarray) -> bool:
        """
        输入音频块（如 320/640/960 点，20/40/60ms），累积后进行关键词检测
        返回是否检测到唤醒词（一旦检测到，后续调用将直接返回 True）
        """
        if self.is_keyword_detected:
            return True

        if chunk.dtype not in (np.float32, np.int16):
            raise ValueError("音频块必须是 float32 或 int16")

        # 累积到缓冲区
        self._buffer.append(chunk)
        self._total_samples += len(chunk)
        self._trim_buffer()

        # 获取当前完整音频用于检测
        audio_data = self._get_buffered_audio()
        if audio_data is None or len(audio_data) < 800:  # 至少 50ms
            return False

        # 调用 KWS 检测（同步）
        try:
            keyword = self.model.detect_keyword(audio_data)
            if keyword is not None:
                self.is_keyword_detected = True
                debug(f"StreamingKWS: 检测到唤醒词 '{keyword}'")
                return True
            return False
        except Exception as e:
            error(f"流式KWS检测异常: {e}")
            return False

    def is_detected(self) -> bool:
        return self.is_keyword_detected
