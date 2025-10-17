# src\services\state_machine.py

from typing import List
import numpy as np
from ..common import VADState
from .event_handler import EventHandler
import asyncio

class StateMachine:
    def __init__(self, handler: EventHandler):
        self.state = VADState.IDLE
        self.audio_buffer: List[np.ndarray] = []
        self.total_samples = 0
        self.sample_rate = 16000
        self.handler = handler
        self._current_speech_start_ms: int | None = None

    def add_audio_chunk(self, chunk: np.ndarray):
        if chunk.dtype not in (np.float32, np.int16):
            raise ValueError("音频块必须是 float32 或 int16")
        self.audio_buffer.append(chunk)
        self.total_samples += len(chunk)

        # ✅ 关键：在状态为 SPEAKING 时，立即触发 active
        if self.state == VADState.SPEAKING and self.handler.on_voice_active:
            asyncio.create_task(self.handler.on_voice_active(chunk))

    def get_total_duration_ms(self) -> int:
        return int(self.total_samples * 1000 / self.sample_rate)

    async def update_vad_result(self, vad_segments: List[List[int]]):
        current_time_ms = self.get_total_duration_ms()

        for seg in vad_segments:
            if len(seg) != 2:
                continue
            start, end = seg

            # 情况1: 语音开始 [start, -1]
            if start != -1 and end == -1:
                if start < 100 or current_time_ms < start + 150:
                    continue
                if self.state == VADState.IDLE:
                    self._current_speech_start_ms = start
                    self.state = VADState.SPEAKING
                    await self.handler.on_voice_start()

            # 情况2: 语音结束 [start, end] 或 [-1, end]
            elif end != -1:
                actual_start = start if start != -1 else self._current_speech_start_ms
                if actual_start is None:
                    actual_start = max(0, end - 1000)
                if actual_start >= end or actual_start < 100:
                    continue

                await self._handle_complete_speech(actual_start, end)

    async def _handle_complete_speech(self, start_ms: int, end_ms: int):
        full_audio = (
            np.concatenate(self.audio_buffer)
            if self.audio_buffer
            else np.array([], dtype=np.float32)
        )
        self.state = VADState.IDLE
        self._current_speech_start_ms = None
        await self.handler.on_voice_end(full_audio, start_ms, end_ms)
        self._reset_buffer()

    def _reset_buffer(self):
        self.audio_buffer.clear()
        self.total_samples = 0

    async def interrupt(self):
        if self.state == VADState.SPEAKING:
            self.state = VADState.IDLE
            self._current_speech_start_ms = None
            self._reset_buffer()
            # 注意：这里不调 handler.on_vad_interrupt，除非你有特殊需求

    def reset(self):
        self._reset_buffer()
        self.state = VADState.IDLE
        self._current_speech_start_ms = None
