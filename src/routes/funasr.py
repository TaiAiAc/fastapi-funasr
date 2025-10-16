# src/routes/funasr.py

import uuid
import numpy as np
from typing import Dict
from fastapi import APIRouter, WebSocket

from ..utils import info, debug, error, AudioConverter
from ..services import vad_service, StateMachine, EventHandler

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
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "start":
                state_machine.reset()
                if stream:
                    stream.reset()
                await handler.send_info("开始接收音频")
                info("客户端开始发送音频")

            elif msg_type == "audio":
                try:
                    audio_list = data.get("data")
                    if not isinstance(audio_list, list):
                        raise ValueError("audio data must be a list of floats")

                    audio_chunk = np.array(audio_list, dtype=np.float32)
                    if not (
                        -1.0 <= audio_chunk.min() <= 1.0
                        and -1.0 <= audio_chunk.max() <= 1.0
                    ):
                        raise ValueError("音频数据超出 [-1, 1] 范围")
                    if len(audio_chunk) % 160 != 0:
                        raise ValueError("音频块长度需为160的倍数（10ms对齐）")

                    state_machine.add_audio_chunk(audio_chunk)
                    audio_int16 = AudioConverter.to_int16(audio_chunk)

                    rms = np.sqrt(np.mean(audio_chunk**2))

                    vad_segments = stream.process(audio_int16) if stream else []

                    if vad_segments:
                        await state_machine.update_vad_result(vad_segments)

                    # debug(
                    #     f"""RMS: {rms:.6f}, len: {len(audio_chunk)} ,采样率: {state_machine.sample_rate}, min/max: {audio_chunk.min():.6f} ~ {audio_chunk.max():.6f},
                    #     VAD segments: {vad_segments}"""
                    # )

                except Exception as e:
                    error(f"处理音频失败: {e}")
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif msg_type in ("stop", "reset"):
                state_machine.reset()
                if stream:
                    stream.reset()
                await handler.send_info("已重置")
                if msg_type == "stop":
                    break

            else:
                await handler.send_warning("未知消息类型")

    except Exception as e:
        error(f"WebSocket 错误: {e}")
        await websocket.close()
    finally:
        active_sessions.pop(session_id, None)
        active_handlers.pop(session_id, None)
        if websocket.client_state == 1:
            await websocket.close()
        info("WebSocket 连接已关闭")
