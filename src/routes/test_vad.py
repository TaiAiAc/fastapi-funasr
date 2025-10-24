# src\routes\test_vad.py

import json
import numpy as np
from fastapi import APIRouter, WebSocket

from ..utils import info, debug, error
from ..services.vad import vad_service

test_vad_router = APIRouter(
    prefix="/test_vad",
    tags=["WebSocket语音识别"],
)


class AudioBuffer:
    def __init__(self, target_len=1600,overlap_len=320):  # 100ms @16kHz
        self.buf = np.array([], dtype=np.int16)
        self.target = target_len
        self.overlap = overlap_len

    def add(self, audio: np.ndarray):
        self.buf = np.concatenate([self.buf, audio.astype(np.int16)])

    def get_chunks(self):
        chunks = []
        while len(self.buf) >= self.target:
            chunks.append(self.buf[: self.target])
            self.buf = self.buf[self.target :]
        return chunks

    def clear(self):
        self.buf = np.array([], dtype=np.int16)

    @property
    def current_duration_ms(self):
        """当前缓冲区中的音频时长（毫秒）"""
        return len(self.buf) / 16  # 16kHz采样率，每毫秒16个采样点

    def get_remaining(self):
        """获取缓冲区中剩余的数据（不足一个完整块时）"""
        if len(self.buf) > 0:
            return self.buf.copy()
        return np.array([], dtype=np.int16)


@test_vad_router.websocket("/ws")
async def websocket_asr(websocket: WebSocket):
    await websocket.accept()
    client_id = id(websocket)
    stream = vad_service.create_stream()
    debug(f"Client {client_id} connected")
    # 初始化音频缓冲区
    audio_buffer = AudioBuffer()  # 100ms

    try:
        while True:
            # 👇 统一接收入口
            message = await websocket.receive()

            # 情况1：收到二进制音频数据
            if "bytes" in message:
                data = message["bytes"]

                audio_int16 = np.frombuffer(data, dtype=np.int16)
                audio_buffer.add(audio_int16)

                chunks = audio_buffer.get_chunks()
 
                for chunk in chunks:
                    result = stream.process(chunk) 
                    if result:
                        await websocket.send_json(
                            {"type": "vad_event", "payload": {"event": "voice_start"}}
                        )

            # 情况2：收到文本/JSON 控制消息
            elif "text" in message:
                payload = json.loads(message["text"])
                msg_type = payload.get("type")

                if msg_type == "stop":
                    result = stream.finish()
                    await websocket.send_json(
                        {"type": "asr_result", "payload": {"final": result}}
                    )
                    break

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

                else:
                    await websocket.send_json(
                        {"type": "error", "msg": "unknown command"}
                    )

    finally:
        audio_buffer.clear()
        stream = None
