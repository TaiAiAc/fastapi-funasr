# src/routes/funasr.py

import json
import uuid
import numpy as np
from typing import Dict
from fastapi import APIRouter, WebSocket

from ..utils import info, debug, error, AudioConverter
from ..services import vad_service, VADSession, SessionHandler

websocket_router = APIRouter(
    prefix="/funasr",
    tags=["WebSocket语音识别"],
)

active_sessions: Dict[str, VADSession] = {}
active_handlers: Dict[str, SessionHandler] = {}  # 可选：便于调试


@websocket_router.websocket("/ws")
async def websocket_asr(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    stream = None

    info(f"WebSocket 连接已建立，会话ID: {session_id}")

    try:
        if not vad_service.is_initialized():
            await websocket.send_text(
                json.dumps(
                    {"type": "warning", "message": "VAD模型未初始化，语音检测不可用"}
                )
            )

        # ✅ 创建会话处理器
        handler = SessionHandler(websocket, session_id)
        active_handlers[session_id] = handler

        # ✅ 创建 VADSession，直接绑定 handler 的方法（无 lambda！）
        vad_session = VADSession(handler=handler)
        active_sessions[session_id] = vad_session

        if vad_service.is_initialized():
            stream = vad_service.create_stream(
                max_end_silence_time=600, speech_noise_thres=0.8
            )

        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "start":
                vad_session.reset()
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

                    vad_session.add_audio_chunk(audio_chunk)
                    audio_int16 = AudioConverter.to_int16(audio_chunk)

                    rms = np.sqrt(np.mean(audio_chunk**2))
                    debug(f"Audio chunk RMS: {rms:.6f}, len: {len(audio_chunk)}")
                    debug(
                        f"Audio chunk min/max: {audio_chunk.min():.6f} ~ {audio_chunk.max():.6f}"
                    )

                    vad_segments = stream.process(audio_int16) if stream else []
                    debug(f"VAD segments: {vad_segments}")
                    if vad_segments:
                        await vad_session.update_vad_result(vad_segments)

                except Exception as e:
                    error(f"处理音频失败: {e}")
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif msg_type in ("stop", "reset"):
                vad_session.reset()
                if stream:
                    stream.reset()
                await handler.send_info("已重置")
                if msg_type == "stop":
                    break

            else:
                await handler.send_warning("未知消息类型")

    except Exception as e:
        error(f"WebSocket 错误: {e}")
    finally:
        active_sessions.pop(session_id, None)
        active_handlers.pop(session_id, None)
        if websocket.client_state == 1:
            await websocket.close()
        info("WebSocket 连接已关闭")
