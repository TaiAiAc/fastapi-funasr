# src/services/event_handler.py

from enum import Enum
import time
import numpy as np
from fastapi import WebSocket
from .kws import kws_service
from .asr import asr_service
from ..utils import debug, error, info, warning, AudioConverter
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import WebSocketDisconnect


class ASRAudioBuffer:
    def __init__(self, target_len=9600):
        self.buf = np.array([], dtype=np.float32)
        self.target = target_len

    def add(self, audio: np.ndarray):
        self.buf = np.concatenate([self.buf, audio.astype(np.float32)])

    def get_chunks(self):
        chunks = []
        while len(self.buf) >= self.target:
            chunks.append(self.buf[: self.target])
            self.buf = self.buf[self.target :]
        return chunks

    def clear(self):
        self.buf = np.array([], dtype=np.float32)


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
        self._asr_stream = None
        # 全局线程池
        self._executor = ThreadPoolExecutor(max_workers=4)
        self.is_bot_speaking = False
        self.audio_buffer = ASRAudioBuffer()

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def set_bot_speaking(self, speaking: bool):
        self.is_bot_speaking = speaking
        debug(f"[{self.session_id}] 数字人说话状态更新: {speaking}")

    async def _send(self, msg_type: str, payload: dict):
        """统一发送格式"""

        try:
            await self.websocket.send_json(
                {
                    "type": msg_type,
                    "payload": payload or {},
                    "timestamp": self._now_ms(),
                    "session_id": self.session_id,
                }
            )
        except (WebSocketDisconnect, RuntimeError) as e:
            warning(f"WebSocket 已断开或发送失败: {e}")

    async def on_voice_start(self):
        if self._state != SessionState.IDLE:
            return
        self._state = SessionState.SPEAKING
        await self._send("vad_event", {"event": "voice_start"})

    async def on_voice_active(self, chunk: np.ndarray):
        chunk_float32 = AudioConverter.int16_to_float32(chunk)
        self.audio_buffer.add(chunk_float32)

        if self.is_bot_speaking:
            # 数字人正在说话 → 启用 KWS 检测唤醒词
            has_keyword = self.kws_service.process_chunk(chunk_float32)
            if has_keyword:
                await self._trigger_wakeup()
            return

        # 数字人没在说话 → 直接走 ASR（无需唤醒）
        self._state = SessionState.SPEAKING

        try:
            chunks = self.audio_buffer.get_chunks()

            if not self._asr_stream:
                self._asr_stream = self.asr_service.create_stream()

            for chunk in chunks:
                partial_text = self._asr_stream.feed_chunk(chunk)
                info(f"[{self.session_id}] ASR 部分识别结果: {partial_text}")
                await self._send("asr_event", {"partial": partial_text})

        except Exception as e:
            error(f"ASR 流式识别异常: {e}")

    async def _trigger_wakeup(self):
        if self._state == SessionState.WAKEUP:
            # 打断：结束当前 ASR，新建流
            self._asr_stream.reset()
            await self._send("interrupt")
        else:
            self._state = SessionState.WAKEUP
            await self._send("wakeup")

    async def on_voice_end(self):
        try:
            if self._state == SessionState.WAKEUP:
                final_text = ""
                try:
                    final_text = await asyncio.get_event_loop().run_in_executor(
                        self._executor, self._asr_stream.finalize
                    )
                except Exception as e:
                    error(f"ASR finalize 异常: {e}")
                await self._send("asr_result", {"final": final_text})
            else:
                # SPEAKING 状态（未唤醒）也通知客户端：语音已结束
                await self._send("vad_event", {"event": "voice_end"})
        finally:
            self._state = SessionState.IDLE
            self._reset()

    async def on_vad_interrupt(self):
        if self._state == SessionState.SPEAKING:
            await self._send("interrupt")
        self._reset()

    async def _reset(self):
        # 确保 finalize 被调用（即使异常）
        try:
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._asr_stream.finalize
            )
        except:
            pass
        self.audio_buffer.clear()
        self._asr_stream = None

    # 可选：暴露 info/warning/error 给路由层调用
    async def send_info(self, message: str):
        await self._send("info", {"message": message})

    async def send_warning(self, message: str):
        await self._send("warning", {"message": message})

    async def send_error(self, message: str):
        await self._send("error", {"message": message})
