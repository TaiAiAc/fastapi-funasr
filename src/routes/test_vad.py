# src\routes\test_asr.py

import json
import numpy as np
from typing import Dict
from fastapi import APIRouter, WebSocket

from ..utils import info, debug, error
from ..services.vad import vad_service
from fastapi import WebSocketDisconnect
import asyncio

test_vad_router = APIRouter(
    prefix="/test_vad",
    tags=["WebSocket语音识别"],
)


@test_vad_router.websocket("/ws")
async def websocket_asr(websocket: WebSocket):
    await websocket.accept()
    client_id = id(websocket)
    stream = vad_service.create_stream()
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

                result = stream.process(audio_array)
                if result:
                    await websocket.send_json(
                        {"type": "vad_result", "payload": {"result": result}}
                    )
            # 情况2：收到文本/JSON 控制消息
            elif "text" in message:
                try:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type")

                    if msg_type == "stop":
                        # 处理结束
                        remaining = buffers[client_id].buf
                        if len(remaining) > 0:
                            result = stream.finalize(remaining)
                        else:
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
        buffers.pop(client_id, None)
        stream = None
