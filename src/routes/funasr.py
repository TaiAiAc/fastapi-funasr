# src/routes/funasr.py

import uuid
import numpy as np
from typing import Dict
from fastapi import APIRouter, WebSocket

from ..common import VADState
from ..utils import info, debug, error, AudioSessionRecorder
from ..services import vad_service, StateMachine, EventHandler
from fastapi import WebSocketDisconnect
import json

websocket_router = APIRouter(
    prefix="/funasr",
    tags=["WebSocket语音识别"],
)

active_sessions: Dict[str, StateMachine] = {}
active_handlers: Dict[str, EventHandler] = {}


@websocket_router.websocket("/ws")
async def websocket_asr(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    stream = None
    recorder = AudioSessionRecorder(session_id)

    info(f"WebSocket 连接已建立，会话ID: {session_id}")

    try:
        # ✅ 创建会话处理器
        handler = EventHandler(websocket, session_id)
        active_handlers[session_id] = handler

        state_machine = StateMachine(handler=handler)
        active_sessions[session_id] = state_machine

        if vad_service.is_initialized:
            stream = vad_service.create_stream()

        while True:
            message = await websocket.receive()

            if "bytes" in message:
                data = message["bytes"]
                if len(data) % 4 != 0:
                    print("Invalid audio data length")
                    continue

                # audio_array = np.frombuffer(data, dtype=np.float32)
                audio_array = np.frombuffer(data, dtype=np.int16)
                recorder.add_chunk(audio_array)

                # 1. 先加音频到状态机（用于后续 KWS/ASR）
                state_machine.add_audio_chunk(audio_array)

                # ✅ 关键：在状态为 SPEAKING 时，立即触发 active
                if state_machine.state == VADState.SPEAKING:
                    await handler.on_voice_active(audio_array)

                vad_segments = stream.process(audio_array) if stream else []

                if vad_segments:
                    await state_machine.update_vad_result(vad_segments)

                # 每次都检查是否超时
                await state_machine.check_silence_timeout()

            elif "text" in message:
                payload = json.loads(message["text"])
                msg_type = payload.get("type")

                if msg_type in ("stop", "reset"):
                    state_machine.reset()
                    if stream:
                        stream.reset()
                    await handler.send_info("已重置")
                    if msg_type == "stop":
                        recorder.finalize()
                        break

                else:
                    await handler.send_warning("未知消息类型")

    except WebSocketDisconnect:
        info("客户端主动断开 WebSocket")

    except Exception as e:
        error(f"WebSocket 错误: {e}")
        await websocket.close()

    finally:
        if stream:
            try:
                stream.finish()  # ← 新增：正确结束流
            except Exception as e:
                error(f"VAD stream finish error: {e}")

        active_sessions.pop(session_id, None)
        active_handlers.pop(session_id, None)
        recorder.finalize()
        await websocket.close()
        info("WebSocket 连接已关闭")
