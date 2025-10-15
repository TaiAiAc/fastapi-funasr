# src/services/vad/audio_buffer.py

import numpy as np
from collections import deque
from typing import Optional

class SlidingAudioBuffer:
    def __init__(self, sample_rate: int = 16000, max_duration_sec: float = 5.0):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration_sec)
        self.buffer = deque(maxlen=self.max_samples)
        self.total_pushed = 0  # 全局样本计数（用于时间对齐）

    def push(self, chunk: np.ndarray):
        if chunk.dtype != np.float32:
            raise ValueError("Only float32 audio supported")
        self.buffer.extend(chunk)
        self.total_pushed += len(chunk)

    def get_all(self) -> np.ndarray:
        return np.array(self.buffer, dtype=np.float32)

    def get_latest_chunk(self, num_samples: int) -> Optional[np.ndarray]:
        if len(self.buffer) < num_samples:
            return None
        # 从 deque 末尾取
        data = []
        start_idx = len(self.buffer) - num_samples
        for i in range(start_idx, len(self.buffer)):
            data.append(self.buffer[i])
        return np.array(data, dtype=np.float32)

    def clear(self):
        self.buffer.clear()
        self.total_pushed = 0

    def total_samples_pushed(self) -> int:
        return self.total_pushed

    def current_duration_sec(self) -> float:
        return len(self.buffer) / self.sample_rate

    def get_last_duration(self, duration_sec: float) -> Optional[np.ndarray]:
        """获取最近 duration_sec 秒的音频"""
        if self.total_samples_pushed() == 0:
            return None
        target_samples = int(duration_sec * self.sample_rate)
        return self.get_latest_chunk(target_samples)