# src/services/vad/state_machine.py

from typing import Optional, Callable, Awaitable, List
import numpy as np
from ...common import InteractionState
from .audio_buffer import SlidingAudioBuffer 


class VADStateMachine:
    def __init__(
        self,
        sample_rate: int = 16000,
        min_speech_ms: int = 200,
        max_speech_ms: int = 15000,
        trailing_silence_ms: int = 600,
        buffer_duration_sec: float = 5.0,  
    ):
        self.state = InteractionState.IDLE
        self.audio_buffer = SlidingAudioBuffer(sample_rate, buffer_duration_sec)
        self.sample_rate = sample_rate

        self.speech_start_ms: Optional[int] = None
        self.last_voice_ms: Optional[int] = None

        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.trailing_silence_ms = trailing_silence_ms

        # 回调（全部异步）
        self.on_vad_start: Optional[Callable[[], Awaitable[None]]] = None
        self.on_kws_feed: Optional[Callable[[np.ndarray], Awaitable[None]]] = None
        self.on_kws_wakeup: Optional[Callable[[], Awaitable[None]]] = None
        self.on_asr_feed: Optional[Callable[[np.ndarray], Awaitable[None]]] = None
        self.on_asr_end: Optional[Callable[[np.ndarray], Awaitable[None]]] = None
        self.on_interrupt: Optional[Callable[[], Awaitable[None]]] = None

    def add_audio_chunk(self, chunk: np.ndarray):
        """外部调用：添加音频块（float32）"""
        self.audio_buffer.push(chunk)

    def get_current_time_ms(self) -> int:
        return int(self.audio_buffer.total_samples_pushed() * 1000 / self.sample_rate)

    async def _enter_vad_active(self, start_ms: int):
        self.speech_start_ms = start_ms
        self.last_voice_ms = start_ms
        self.state = InteractionState.VAD_ACTIVE
        if self.on_vad_start:
            await self.on_vad_start()

    async def _force_end(self):
        self.state = InteractionState.IDLE
        self.audio_buffer.clear()
        self.speech_start_ms = None
        self.last_voice_ms = None

    # src/services/vad/state_machine.py

    async def on_vad_segments(self, segments: List[List[int]]):
        """接收 FunASR VAD 的原始 segments，如 [[1000, -1], [-1, 2000]]"""
        current_time_ms = self.get_current_time_ms()

        for seg in segments:
            if len(seg) != 2:
                continue
            start, end = seg

            # 情况1: 语音开始 [start, -1]
            if start != -1 and end == -1:
                if self.state == InteractionState.IDLE:
                    await self._enter_vad_active(start)

            # 情况2: 语音结束 [-1, end] 或 [start, end]
            elif end != -1:
                if self.state in (
                    InteractionState.VAD_ACTIVE,
                    InteractionState.ASR_ACTIVE,
                ):
                    # 触发 ASR 结束（可选）
                    if self.on_asr_end:
                        final_audio = self.audio_buffer.get_all()
                        await self.on_asr_end(final_audio)
                    await self._force_end()

    async def trigger_kws_wakeup(self):
        """由 KWS 服务调用：检测到唤醒词"""
        if self.state in (InteractionState.VAD_ACTIVE, InteractionState.ASR_ACTIVE):
            self.state = InteractionState.INTERRUPTING
            if self.on_interrupt:
                await self.on_interrupt()

            # 切换到 ASR 模式，并送入当前缓冲区全部音频（简化）
            self.state = InteractionState.ASR_ACTIVE
            if self.on_asr_end:
                final_audio = self.audio_buffer.get_all()
                await self.on_asr_end(final_audio)

    def reset(self):
        self.state = InteractionState.IDLE
        self.audio_buffer.clear()
        self.speech_start_ms = None
        self.last_voice_ms = None
