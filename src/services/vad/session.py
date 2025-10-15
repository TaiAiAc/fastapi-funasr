from typing import Optional, Callable, Awaitable, List
from ...common import VADState

class VADSession:
    def __init__(
        self,
        on_voice_start: Optional[Callable[[], None]] = None,
        on_voice_end: Optional[Callable[[List[bytes], float, float], Awaitable[None]]] = None,
    ):
        self.state = VADState.IDLE
        self.audio_buffer: List[bytes] = []
        self.total_audio_ms = 0  # 已接收音频总时长（毫秒）
        self.chunk_duration_ms = 30  # 假设每 chunk 30ms，根据你的采样率调整

        self.on_voice_start = on_voice_start
        self.on_voice_end = on_voice_end

        self._current_speech_start_ms: Optional[int] = None
        self._pending_segments = []  # 存储带 -1 的段落

    def add_audio_chunk(self, chunk: bytes):
        """添加原始音频 chunk（用于后续 ASR）"""
        self.audio_buffer.append(chunk)
        self.total_audio_ms += self.chunk_duration_ms

    def update_vad_result(self, vad_segments: List[List[int]]):
        """
        处理 FunASR VAD 返回的 segments，如 [[18650, -1], [-1, 19680]]
        """
        # 合并 pending 和新 segments
        all_segments = self._merge_segments(self._pending_segments, vad_segments)
        self._pending_segments = []

        # 分离完整段和未完成段
        complete_segments = []
        incomplete_segments = []

        for seg in all_segments:
            start, end = seg
            if start != -1 and end != -1:
                complete_segments.append((start, end))
            else:
                incomplete_segments.append(seg)

        # 保存未完成的
        self._pending_segments = incomplete_segments

        # 处理完整语音段
        for start_ms, end_ms in complete_segments:
            self._handle_complete_speech(start_ms, end_ms)

        # 检查是否有新的语音开始（有 start!=-1 但 end=-1）
        for seg in incomplete_segments:
            if seg[0] != -1 and self.state == VADState.IDLE:
                self._on_voice_start(seg[0])

    def _merge_segments(self, pending, new_segments):
        """合并 pending 和新 segments，尝试补全"""
        # 简单策略：把 new_segments 追加，FunASR 通常会修正之前的 -1
        merged = pending + new_segments
        # 去重或合并逻辑可选（FunASR 一般不会重复）
        return merged

    def _on_voice_start(self, start_ms: int):
        if self.state == VADState.IDLE:
            self.state = VADState.SPEAKING
            self._current_speech_start_ms = start_ms
            if self.on_voice_start:
                self.on_voice_start()

    def _handle_complete_speech(self, start_ms: int, end_ms: int):
        # 注意：这段语音可能早已开始，现在才确认结束
        if self.state in (VADState.SPEAKING, VADState.IDLE):
            self.state = VADState.VOICE_END

            # 计算需要截取的音频范围（简化版：假设 buffer 从 0 开始）
            # 更精确做法：用时间戳对齐，但这里用总 buffer 长度近似
            # 假设你只关心最后一段语音
            audio_data = self.audio_buffer.copy()

            if self.on_voice_end:
                # 异步处理，注意：不要阻塞
                import asyncio
                asyncio.create_task(
                    self.on_voice_end(audio_data, start_ms, end_ms)
                )

            # 重置（或保留部分 buffer 用于下一段？）
            self.audio_buffer.clear()
            self.state = VADState.IDLE
            self._current_speech_start_ms = None

    def reset(self):
        self.state = VADState.IDLE
        self.audio_buffer.clear()
        self._pending_segments.clear()
        self._current_speech_start_ms = None