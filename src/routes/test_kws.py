# src/routes/test_asr.py

import json
import numpy as np
from typing import Dict
from fastapi import APIRouter, WebSocket

from ..utils import info, debug, error, log_audio_input, AudioSessionRecorder
from ..services.kws import kws_service  # ← 新封装
import uuid

test_kws_router = APIRouter(
    prefix="/test_kws",
    tags=["WebSocket关键词识别"],
)


@test_kws_router.websocket("/ws")
async def websocket_kws(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    recorder = AudioSessionRecorder(session_id=session_id)

    debug(f"Client {session_id} connected")

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                data = message["bytes"]
                if len(data) % 4 != 0:
                    error("Invalid audio data length")
                    continue

                # 假设前端发送的是 float32 PCM, 16kHz, 单声道
                audio_array = np.frombuffer(data, dtype=np.float32)
                # 注意：这里每次收到的是一个 chunk（比如 960/1600/3200 samples）

                log_audio_input(
                    audio=audio_array,
                    name="KWS",
                    sample_rate=16000,
                    expected_format="float32",  # ← 关键！
                )

                recorder.add_chunk(audio_array)

                # 直接送入流式 KWS
                result = kws_service.process_chunk(audio_array)

                if result and result["detected"]:
                    await websocket.send_json(
                        {
                            "type": "kws_event",
                            "detected": True,
                            "keyword": result["keyword"],
                            "score": result["score"],
                        }
                    )
                    # 可选：检测到后重置，等待下一次唤醒
                    kws_service.reset()

            elif "text" in message:
                try:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "stop":
                        # 最终处理（可选）
                        result = kws_service.process_chunk(np.array([]), is_final=True)
                        await websocket.send_json({"type": "final", "result": result})
                        break
                    elif payload.get("type") == "reset":
                        kws_service.reset()
                        await websocket.send_json({"type": "reset_ok"})
                except Exception as e:
                    await websocket.send_json({"type": "error", "msg": str(e)})

    except Exception as e:
        error(f"WebSocket error: {e}")
    finally:
        recorder.finalize()
        kws_service.reset()
