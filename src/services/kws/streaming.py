# src/services/kws/streaming.py

from ...utils import debug, error, info
import numpy as np
from typing import Optional
from typing import Dict, Any


class StreamingKWSService:
    def __init__(
        self,
        kws_service,
        generate_options: Dict[str, Any],
        buffer_duration_sec: float = 0.3,  # 缓冲时长从1.5秒降至0.3秒
        sample_rate: int = 16000,
        chunk_size: list = [0, 5, 2],
    ):  # 分块参数调整为300ms
        self.kws_service = kws_service
        self.sample_rate = sample_rate
        self.max_buffer_samples = int(buffer_duration_sec * sample_rate)
        self.reset()

    def reset(self):
        self._buffer = []
        self.cache = {}  # 显式重置cache为空字典
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
        if self.is_keyword_detected:
            return True

        # 转为 float32 [-1, 1]
        if chunk.dtype == np.int16:
            chunk = chunk.astype(np.float32) / 32768.0
        elif chunk.dtype != np.float32:
            raise ValueError(f"Unsupported dtype: {chunk.dtype}")

        try:
            # 调用流式接口
            output = self.kws_service._model.generate(
                **self.generate_options,
                input=chunk,
                cache=self.cache,
                return_cache=True,
            )

            keyword = self.kws_service.parse_kws_result(
                output, threshold=0.1
            )  # 提高阈值至0.3
            if keyword:
                self.is_keyword_detected = True
                self.reset()  # 检测到唤醒词后重置缓冲区
                return True

            return False

        except Exception as e:
            error(f"流式KWS异常: {e}")
            return False

    def is_detected(self) -> bool:
        """是否检测到关键词"""
        return self.is_keyword_detected
