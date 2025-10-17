# src/services/event_handler.py

from enum import Enum
import asyncio
import time
import numpy as np
from fastapi import WebSocket
from .kws import kws_service
from .asr import asr_service
from ..utils import debug, error, info


class SessionState(Enum):
    IDLE = "idle"
    SPEAKING = "speaking"
    WAKEUP = "wakeup"


class EventHandler:
    def __init__(self, websocket: WebSocket, session_id: str):
        self.websocket = websocket
        self.session_id = session_id
        self.kws_service = kws_service
        self.asr_service = asr_service
        self._state = SessionState.IDLE
        self._kws_stream = None
        self._asr_stream = None

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    async def _send(self, msg_type: str, payload: dict = None):
        """统一发送格式"""
        await self.websocket.send_json(
            {
                "type": msg_type,
                "payload": payload or {},
                "timestamp": self._now_ms(),
                "session_id": self.session_id,
            }
        )

    async def on_voice_start(self):
        if self._state != SessionState.IDLE:
            return
        self._state = SessionState.SPEAKING
        self._kws_stream = self.kws_service.create_stream()
        await self._send("vad_event", {"event": "voice_start"})

    async def on_voice_active(self, chunk: np.ndarray):
        if self._state == SessionState.WAKEUP:
            if self._asr_stream:
                try:
                    partial_text = self._asr_stream.feed_chunk(chunk)
                    if partial_text:
                        await self._send("asr_event", {"partial": partial_text})
                except Exception as e:
                    error(f"ASR 流式识别异常: {e}")
            return

        if self._state == SessionState.SPEAKING and self._kws_stream:
            try:
                if self._kws_stream.detect_keyword_stream(chunk):
                    await self._trigger_wakeup()
            except Exception as e:
                error(f"KWS 流式检测异常: {e}")

    async def _trigger_wakeup(self):
        if self._state == SessionState.WAKEUP:
            # 打断：结束当前 ASR，新建流
            if self._asr_stream:
                try:
                    self._asr_stream.finalize()
                except Exception as e:
                    error(f"ASR finalize 失败: {e}")
            await self._send("interrupt")
        else:
            self._state = SessionState.WAKEUP
            await self._send("wakeup")

        # 创建新的 ASR 流
        if self.asr_service.is_initialized:
            self._asr_stream = self.asr_service.create_stream()

    async def on_voice_end(self, full_audio: np.ndarray, start_ms: int, end_ms: int):
        try:
            if self._state == SessionState.WAKEUP:
                final_text = ""
                if self._asr_stream:
                    try:
                        final_text = self._asr_stream.finalize()
                    except Exception as e:
                        error(f"ASR finalize 异常: {e}")
                await self._send("asr_result", {"final": final_text})
            # SPEAKING 状态无输出，静默结束
        finally:
            self._state = SessionState.IDLE
            self._reset()

    def _reset(self):
        self._kws_stream = None
        self._asr_stream = None

    # 可选：暴露 info/warning/error 给路由层调用
    async def send_info(self, message: str):
        await self._send("info", {"message": message})

    async def send_warning(self, message: str):
        await self._send("warning", {"message": message})

    async def send_error(self, message: str):
        await self._send("error", {"message": message})
