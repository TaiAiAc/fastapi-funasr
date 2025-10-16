# src/services/websocket_session.py

import json
from typing import Optional
import numpy as np
from fastapi import WebSocket

from ..utils import info, debug, error
from .kws import KWSService
from .asr import ASRService


class EventHandler:
    def __init__(self, websocket: WebSocket, session_id: str):
        self.websocket = websocket
        self.session_id = session_id
        self.kws_service = KWSService()
        self.asr_service = ASRService()
        self._has_interrupted = False  # 可选：记录是否已被打断

    async def on_voice_start(self):
        await self.websocket.send_json(
            {
                "type": "vad_event",
                "event": "voice_start",
                "message": f"会话 {self.session_id} 检测到人声开始",
            }
        )

    async def on_voice_active(self, chunk: np.ndarray, timestamp_ms: int):
        # TODO: 流式 KWS 检测（可选）
        if self.kws_service.is_initialized and not self._has_interrupted:
            # 示例：假设 detect_stream 返回 bool
            # if await self.kws_service.detect_stream(chunk):
            #     await self.on_vad_interrupt()
            pass

        # 流式 ASR（可选）
        # if self.asr_service.is_initialized():
        #     partial = await self.asr_service.recognize_stream(chunk)
        #     if partial:
        #         await self.websocket.send_json({"type": "asr_partial", "text": partial})

    async def on_voice_end(self, full_audio: np.ndarray, start_ms: int, end_ms: int):
        # 可选：用完整音频做最终识别
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
