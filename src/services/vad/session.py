# src\services\vad\session.py

from typing import Optional, Callable, Awaitable, List
import numpy as np
from ...common import VADState

class VADSession:
    def __init__(
        self,
        on_voice_start: Optional[Callable[[], Awaitable[None]]] = None,
        on_voice_active: Optional[Callable[[np.ndarray, int], Awaitable[None]]] = None,
        on_voice_end: Optional[Callable[[np.ndarray, int, int], Awaitable[None]]] = None,
    ):
        self.state = VADState.IDLE
        self.audio_buffer: List[np.ndarray] = []  # 存 float32 或 int16，统一用 np.ndarray
        self.total_samples = 0  # 总采样点数（16kHz）
        self.sample_rate = 16000

        # 回调函数（全部异步）
        self.on_voice_start = on_voice_start      # 语音开始（VAD 触发）
        self.on_voice_active = on_voice_active    # 每次有新语音块（用于 KWS 流式输入）
        self.on_voice_end = on_voice_end          # 语音结束（用于 ASR 最终识别）

        self._speech_start_time_ms: Optional[int] = None
        self._pending_segments: List[List[int]] = []

    def add_audio_chunk(self, chunk: np.ndarray):
        """添加音频块（float32 或 int16），用于后续 KWS/ASR"""
        if chunk.dtype not in (np.float32, np.int16):
            raise ValueError("音频块必须是 float32 或 int16")
        self.audio_buffer.append(chunk)
        self.total_samples += len(chunk)

    def get_total_duration_ms(self) -> int:
        return int(self.total_samples * 1000 / self.sample_rate)

    def update_vad_result(self, vad_segments: List[List[int]]):
        """
        处理 FunASR VAD 返回的 segments，如 [[18650, -1], [-1, 19680]]
        核心：不仅要处理完整段，还要在语音进行中触发 on_voice_active
        """
        # 合并 pending 和新 segments
        all_segments = self._pending_segments + vad_segments
        self._pending_segments = []

        complete_segments = []
        incomplete_segments = []

        for seg in all_segments:
            if len(seg) != 2:
                continue
            start, end = seg
            if start != -1 and end != -1:
                complete_segments.append((start, end))
            else:
                incomplete_segments.append(seg)

        self._pending_segments = incomplete_segments

        # === 处理完整语音段（结束）===
        for start_ms, end_ms in complete_segments:
            self._handle_complete_speech(start_ms, end_ms)

        # === 检查是否有新语音开始（start != -1）===
        for seg in incomplete_segments:
            if seg[0] != -1 and self.state == VADState.IDLE:
                self._on_voice_start(seg[0])

        # === 如果正在说话，触发 on_voice_active（用于 KWS 流式输入）===
        if self.state == VADState.SPEAKING and self.on_voice_active and self.audio_buffer:
            # 传递最新 chunk 和当前时间戳（可选）
            latest_chunk = self.audio_buffer[-1]
            current_time_ms = self.get_total_duration_ms()
            import asyncio
            asyncio.create_task(self.on_voice_active(latest_chunk, current_time_ms))

    def _on_voice_start(self, start_ms: int):
        if self.state == VADState.IDLE:
            self.state = VADState.SPEAKING
            self._speech_start_time_ms = start_ms
            if self.on_voice_start:
                import asyncio
                asyncio.create_task(self.on_voice_start())

    def _handle_complete_speech(self, start_ms: int, end_ms: int):
        if self.state in (VADState.SPEAKING, VADState.IDLE):
            # 合并 buffer 为完整音频
            if self.audio_buffer:
                full_audio = np.concatenate(self.audio_buffer)
            else:
                full_audio = np.array([], dtype=np.float32)

            self.state = VADState.VOICE_END
            if self.on_voice_end:
                import asyncio
                asyncio.create_task(self.on_voice_end(full_audio, start_ms, end_ms))

            # 重置 buffer（注意：若需支持重叠语音，可保留部分）
            self.audio_buffer.clear()
            self._speech_start_time_ms = None
            self.state = VADState.IDLE

    def reset(self):
        self.state = VADState.IDLE
        self.audio_buffer.clear()
        self._pending_segments.clear()
        self._speech_start_time_ms = None
        self.total_samples = 0