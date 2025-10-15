import time
from typing import List, Optional, Callable, Awaitable
from ...common import VADState

class VADStateMachine:
    def __init__(
        self,
        min_speech_duration: float = 0.2,  # 最短有效语音（秒）
        trailing_silence: float = 0.6,  # 语音结束后容忍的静音（秒）
        max_speech_duration: float = 15.0,  # 最长语音（防卡死）
        chunk_duration: float = 0.03,  # 每个 chunk 的时长（秒），如 30ms
    ):
        self.state = VADState.IDLE
        self.buffer: List[bytes] = []
        self.speech_start_time: Optional[float] = None
        self.last_voice_time: Optional[float] = None

        self.min_speech_duration = min_speech_duration
        self.trailing_silence = trailing_silence
        self.max_speech_duration = max_speech_duration
        self.chunk_duration = chunk_duration

        # 回调函数（由外部注入）
        self.on_voice_start: Optional[Callable[[], None]] = None
        self.on_voice_end: Optional[Callable[[List[bytes]], Awaitable[None]]] = None

    def reset(self):
        """重置状态机，用于新会话或异常恢复"""
        self.state = VADState.IDLE
        self.buffer.clear()
        self.speech_start_time = None
        self.last_voice_time = None

    async def update(self, chunk: bytes, is_speech: bool) -> None:
        """
        更新状态机状态
        :param chunk: 原始音频字节（用于缓冲）
        :param is_speech: VAD 判断结果（True/False）
        """
        current_time = time.time()

        if self.state == VADState.IDLE:
            if is_speech:
                self._enter_speaking(current_time, chunk)

        elif self.state == VADState.SPEAKING:
            self.buffer.append(chunk)
            if is_speech:
                self.last_voice_time = current_time
                # 检查是否超时
                if current_time - self.speech_start_time > self.max_speech_duration:
                    await self._force_end_speech(current_time)
            else:
                # 进入尾部静音容忍期
                self.state = VADState.VOICE_END

        elif self.state == VADState.VOICE_END:
            self.buffer.append(chunk)  # 可选：保留尾部静音用于 ASR
            if is_speech:
                # 语音继续
                self.last_voice_time = current_time
                self.state = VADState.SPEAKING
            else:
                # 检查尾部静音是否超时
                if current_time - self.last_voice_time >= self.trailing_silence:
                    await self._finalize_speech()

        elif self.state == VADState.PROCESSING:
            # 当前正在处理，丢弃新输入（不支持打断）
            # 如需支持打断，可在此处重置并重新开始
            pass

    def _enter_speaking(self, current_time: float, first_chunk: bytes):
        self.speech_start_time = current_time
        self.last_voice_time = current_time
        self.buffer = [first_chunk]
        self.state = VADState.SPEAKING
        if self.on_voice_start:
            self.on_voice_start()

    async def _force_end_speech(self, current_time: float):
        """强制结束过长语音"""
        self.last_voice_time = current_time
        await self._finalize_speech()

    async def _finalize_speech(self):
        """确认语音结束，触发后续处理"""
        if self.speech_start_time is None:
            return

        speech_duration = self.last_voice_time - self.speech_start_time
        if speech_duration < self.min_speech_duration:
            # 太短，视为噪声
            self.reset()
            return

        self.state = VADState.PROCESSING

        # 触发异步处理
        if self.on_voice_end:
            # 注意：on_voice_end 必须在完成后调用 self.reset()
            await self.on_voice_end(self.buffer.copy())
        else:
            self.reset()