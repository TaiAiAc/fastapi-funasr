# src/services/vad/session.py
from typing import Optional, Callable, Awaitable, List
import numpy as np
from ...common import VADState  # 注意：这里应只用 VADState，InteractionState 已不用
import asyncio


class VADSession:
    def __init__(
        self,
        on_voice_start: Optional[Callable[[], Awaitable[None]]] = None,
        on_voice_active: Optional[Callable[[np.ndarray, int], Awaitable[None]]] = None,
        on_voice_end: Optional[
            Callable[[np.ndarray, int, int], Awaitable[None]]
        ] = None,
        on_vad_interrupt: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self.state = VADState.IDLE
        self.audio_buffer: List[np.ndarray] = []
        self.total_samples = 0
        self.sample_rate = 16000

        self.on_voice_start = on_voice_start
        self.on_voice_active = on_voice_active
        self.on_voice_end = on_voice_end
        self.on_vad_interrupt = on_vad_interrupt

        self._speech_start_time_ms: Optional[int] = None
        self._pending_segments: List[List[int]] = []

    def add_audio_chunk(self, chunk: np.ndarray):
        if chunk.dtype not in (np.float32, np.int16):
            raise ValueError("音频块必须是 float32 或 int16")
        self.audio_buffer.append(chunk)
        self.total_samples += len(chunk)

    def get_total_duration_ms(self) -> int:
        return int(self.total_samples * 1000 / self.sample_rate)

    async def update_vad_result(self, vad_segments: List[List[int]]):
        current_time_ms = self.get_total_duration_ms()
        all_segments = self._pending_segments + vad_segments
        self._pending_segments = []

        for seg in all_segments:
            if len(seg) != 2:
                continue
            start, end = seg

            # 情况1: 语音开始 [start, -1]
            if start != -1 and end == -1:
                # 🔒 防止过早触发：要求 start 不能太小，且当前时间已超过 start + 一定阈值
                if start < 100:  # 忽略前100ms的“假开始”
                    continue
                if current_time_ms < start + 200:  # 至少等200ms确认
                    # 暂存，稍后再处理（或直接丢弃，简化逻辑）
                    # 这里我们选择：只在 current_time_ms 足够时才触发
                    pass
                if self.state == VADState.IDLE:
                    await self._on_voice_start(start)
                self._pending_segments.append([start, -1])

            # 情况2: 语音结束 [-1, end] 或 [start, end]
            elif end != -1:
                actual_start = start
                if start == -1:
                    for i in range(len(self._pending_segments) - 1, -1, -1):
                        p_start, p_end = self._pending_segments[i]
                        if p_start != -1 and p_end == -1:
                            actual_start = p_start
                            self._pending_segments.pop(i)
                            break
                    else:
                        continue
                if actual_start >= end or actual_start < 100:  # 同样过滤过早语音
                    continue

                if self.state in (VADState.SPEAKING, VADState.IDLE):
                    await self._handle_complete_speech(actual_start, end)

        # 流式活跃回调
        if (
            self.state == VADState.SPEAKING
            and self.on_voice_active
            and self.audio_buffer
        ):
            latest_chunk = self.audio_buffer[-1]
            await self.on_voice_active(latest_chunk, current_time_ms)

    async def _on_voice_start(self, start_ms: int):
        if self.state == VADState.IDLE:
            self.state = VADState.SPEAKING
            self._speech_start_time_ms = start_ms
            if self.on_voice_start:
                await self.on_voice_start()

    async def _handle_complete_speech(self, start_ms: int, end_ms: int):
        if self.state in (VADState.SPEAKING, VADState.IDLE):
            full_audio = (
                np.concatenate(self.audio_buffer)
                if self.audio_buffer
                else np.array([], dtype=np.float32)
            )
            self.state = VADState.VOICE_END
            if self.on_voice_end:
                await self.on_voice_end(full_audio, start_ms, end_ms)
            self._reset_buffer()

    def _reset_buffer(self):
        self.audio_buffer.clear()
        self.total_samples = 0
        self._speech_start_time_ms = None
        self.state = VADState.IDLE

    async def interrupt(self):
        if self.state in (VADState.SPEAKING, VADState.VOICE_END):
            self.state = VADState.IDLE
            self._reset_buffer()
            self._pending_segments.clear()  # 👈 关键：清空 pending segments
            if self.on_vad_interrupt:
                await self.on_vad_interrupt()

    def reset(self):
        self._reset_buffer()
        self._pending_segments.clear()
