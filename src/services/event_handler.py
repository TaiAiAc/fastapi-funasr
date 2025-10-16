# src/services/websocket_session.py

import json
import numpy as np
from fastapi import WebSocket

from ..utils import info, debug, error
from .kws import kws_service
from .asr import asr_service


class EventHandler:
    def __init__(self, websocket: WebSocket, session_id: str):
        self.websocket = websocket
        self.session_id = session_id
        self.kws_service = kws_service
        self.asr_service = asr_service
        self._has_interrupted = False  # 可选：记录是否已被打断
        self._kws_stream = None  # 新增：每个会话独立的 KWS 流
        self._is_wake_up = False

    async def on_voice_start(self):
        # 每次人声开始，重置 KWS 流（避免跨句误触发）
        if self.kws_service.is_initialized:
            self._kws_stream = self.kws_service.create_stream()
        self._is_wake_up = False

        await self.websocket.send_json(
            {
                "type": "vad_event",
                "event": "voice_start",
                "message": f"会话 {self.session_id} 检测到人声开始",
            }
        )

    async def on_voice_active(self, chunk: np.ndarray, timestamp_ms: int):
        # 1. 如果尚未唤醒，尝试 KWS 检测
        if (
            not self._is_wake_up
            and self.kws_service.is_initialized
            and self._kws_stream
        ):
            try:
                flag = self._kws_stream.detect_keyword_stream(chunk)

                info(f"会话 {self.session_id} 尝试 KWS 检测，时间戳: {timestamp_ms}ms 检测结果: {flag}")

                if flag:
                    self._is_wake_up = True
                    await self.on_vad_interrupt()  # 触发打断
                    return
            except Exception as e:
                error(f"KWS 流式检测异常: {e}")

        # 2. 如果已唤醒，可送入 ASR（后续实现）
        # if self._is_wake_up and self.asr_service.is_initialized:
        #     partial = await self.asr_service.recognize_stream(chunk)
        #     if partial:
        #         await self.send_asr_partial(partial)

    async def on_voice_end(self, full_audio: np.ndarray, start_ms: int, end_ms: int):
        # 可选：用完整音频做最终识别
        self._has_interrupted = True
        self._is_wake_up = True  # 标记已唤醒
        final_text = ""
        if self.asr_service.is_initialized:
            # final_text = await self.asr_service.finalize(full_audio)
            pass

        await self.websocket.send_json(
            {"type": "vad_event", "event": "asr_final", "message": "语音识别结束"}
        )

    async def on_vad_interrupt(self):
        self._has_interrupted = True
        await self.asr_service.interrupt()
        await self.websocket.send_json(
            {"type": "interrupt", "message": "检测到唤醒词，已打断当前识别"}
        )

    async def send_warning(self, message: str):
        await self.websocket.send_text(
            json.dumps({"type": "warning", "message": message})
        )

    async def send_info(self, message: str):
        await self.websocket.send_json({"type": "info", "message": message})
