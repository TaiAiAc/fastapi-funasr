# src\routes\test_asr.py

import json
import numpy as np
from typing import Dict
from fastapi import APIRouter, WebSocket

from ..utils import info, debug, error, AudioConverter
from ..services.asr import asr_service
from fastapi import WebSocketDisconnect
import asyncio

test_asr_router = APIRouter(
    prefix="/test_asr",
    tags=["WebSocket语音识别"],
)


class AudioBuffer:
    def __init__(self, target_len=9600):
        self.buf = np.array([], dtype=np.float32)
        self.target = target_len

    def add(self, audio: np.ndarray):
        self.buf = np.concatenate([self.buf, audio.astype(np.float32)])

    def get_chunks(self):
        chunks = []
        while len(self.buf) >= self.target:
            chunks.append(self.buf[:self.target])
            self.buf = self.buf[self.target:]
        return chunks

    def clear(self):
        self.buf = np.array([], dtype=np.float32)




@test_asr_router.websocket("/ws")
async def websocket_asr(websocket: WebSocket):
    await websocket.accept()
    client_id = id(websocket)
    buffers = AudioBuffer()
    stream = asr_service.create_stream()
    debug(f"Client {client_id} connected")

    try:
        while True:
            # 👇 统一接收入口
            message = await websocket.receive()

            # 情况1：收到二进制音频数据
            if "bytes" in message:
                data = message["bytes"]
                if len(data) % 4 != 0:
                    print("Invalid audio data length")
                    continue

                audio_array = np.frombuffer(data, dtype=np.float32)
                buffers.add(audio_array)

                # 处理 ASR
                for chunk in buffers.get_chunks():
                    result = stream.feed_chunk(chunk)
                    info(f"Received chunk, result: {result}")
                    if result:
                        await websocket.send_json(
                            {"type": "asr_event", "payload": {"partial": result}}
                        )

            # 情况2：收到文本/JSON 控制消息
            elif "text" in message:
                try:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type")

                    if msg_type == "stop":
                        # 处理结束
                        result = stream.finalize()
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

                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "msg": "invalid JSON"})

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        buffers.clear()
        stream = None
