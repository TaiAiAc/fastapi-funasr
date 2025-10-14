import numpy as np
from typing import List
from ..common import VADState

class VADStream:
    def __init__(self, model):
        """初始化VAD流处理器"""
        self.model = model
        self.reset()

    def reset(self):
        """重置流状态"""
        self.cache = {}
        self.total_samples = 0
        self.chunk_size = 200
        self._last_vad_state = 0

    def process(self, audio_chunk: np.ndarray) -> List[List[int]]:
        """
        处理音频块（流式VAD中间步骤，通常不返回语音段）
        """
        if len(audio_chunk) == 0:
            return []

        # 确保音频在 [-1, 1] 范围内（FunASR 要求）
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        if np.abs(audio_chunk).max() > 1.0:
            # 自动归一化（可选，根据你的数据源决定）
            audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))

        # 流式推理：is_final=False
        result = self.model.generate(
            input=audio_chunk,
            cache=self.cache,
            chunk_size=self.chunk_size,      # 毫秒
            is_final=False
        )

        # 注意：fsmn-vad 在 is_final=False 时通常返回空列表！
        segments_ms = result[0].get("value", [])
        self._last_vad_state = 1 if segments_ms else 0

        # 更新总采样点（用于估算时长）
        self.total_samples += len(audio_chunk)

        # 调试打印
        if segments_ms:
            print(f"🟡 process() 中检测到段（罕见）: {segments_ms}")
        return segments_ms  # 通常为空

    def finish(self) -> List[List[int]]:
        """
        结束流式处理，获取最终语音段列表（毫秒）
        """
        # 发送空输入 + is_final=True 触发最终输出
        result = self.model.generate(
            input=np.array([], dtype=np.float32),
            cache=self.cache,
            chunk_size=self.chunk_size,
            is_final=True
        )

        segments_ms = result[0].get("value", [])
        self._last_vad_state = 1 if segments_ms else 0

        # 安全过滤：确保 start < end，且时间合理
        filtered_segments = []
        estimated_duration_ms = int(self.total_samples / 16.0)  # 16kHz

        for start, end in segments_ms:
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                start, end = int(start), int(end)
                if 0 <= start < end <= estimated_duration_ms + 1000:  # 容忍1秒误差
                    filtered_segments.append([start, end])
                else:
                    print(f"⚠️ 跳过异常段: [{start}, {end}], 音频估计时长: {estimated_duration_ms}ms")
            else:
                print(f"⚠️ 跳过非数值段: {start}, {end}")

        return filtered_segments

    def is_speech_active(self) -> bool:
        return self._last_vad_state == 1

    def get_voice_state(self) -> VADState:
        return VADState.SPEAKING if self._last_vad_state == 1 else VADState.IDLE