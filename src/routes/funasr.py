import json
import uuid
import numpy as np
from typing import Dict
from fastapi import APIRouter, WebSocket

from ..common import VADState
from ..utils import info, error
from ..services.vad import vad_service
from ..services.vad_stream import VADStream
from ..utils.audio_converter import AudioConverter

# 创建WebSocket相关的路由器
websocket_router = APIRouter(
    prefix="/funasr",
    tags=["WebSocket语音识别"],
    responses={404: {"description": "Not found"}},
)

# 管理多个客户端的流
active_streams: Dict[str, VADStream] = {}
session_states: Dict[str, VADState] = {}
audio_buffers: Dict[str, list] = {}  # 缓存 float32 音频用于后续 ASR


@websocket_router.websocket("/ws")
async def websocket_asr(websocket: WebSocket):
    """
    WebSocket语音识别接口
    用于进行语音端点检测(VAD)
    支持检测人声开始和结束节点
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    stream = None

    info(f"WebSocket 连接已建立，会话ID: {session_id}")

    try:
        if not vad_service.is_initialized():
            await websocket.send_text(
                json.dumps({
                    "type": "warning",
                    "message": "VAD模型尚未初始化完成，语音端点检测功能将不可用",
                })
            )
            info("WebSocket连接建立但VAD模型未初始化")
        else:
            stream = vad_service.create_stream()
            active_streams[session_id] = stream
            audio_buffers[session_id] = []
            session_states[session_id] = VADState.IDLE

        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "start":
                session_states[session_id] = VADState.IDLE
                audio_buffers[session_id] = []
                await websocket.send_text(
                    json.dumps({"status": "started", "message": "开始接收音频"})
                )
                info("客户端开始发送音频")

            elif msg_type == "audio":
                if stream is None:
                    await websocket.send_text(json.dumps({"error": "VAD模型未初始化"}))
                    continue

                try:
                    audio_list = data.get("data")
                    if not isinstance(audio_list, list):
                        raise ValueError("audio data must be a list of floats")

                    audio_chunk = np.array(audio_list, dtype=np.float32)
                    if not (-1.0 <= audio_chunk.min() <= 1.0 and -1.0 <= audio_chunk.max() <= 1.0):
                        raise ValueError("音频数据超出 [-1, 1] 范围")

                    if len(audio_chunk) % 160 != 0:
                        raise ValueError(f"音频块长度 {len(audio_chunk)} 不是160的倍数（10ms对齐）")

                    # 转换为 int16 供 VAD 使用
                    audio_chunk_int16 = AudioConverter.to_int16(audio_chunk)

                    # 流式处理
                    stream.process(audio_chunk_int16)
                    is_speaking = stream.is_speech_active()
                    current_state = session_states.get(session_id, VADState.IDLE)

                    # === 状态机逻辑 ===
                    if current_state == VADState.IDLE and is_speaking:
                        # 开始说话
                        audio_buffers[session_id] = [audio_chunk]
                        session_states[session_id] = VADState.SPEAKING
                        await websocket.send_json({
                            "type": "vad_event",
                            "event": "voice_start",
                            "message": "检测到人声开始"
                        })
                        info(f"会话 {session_id} 开始说话")

                    elif current_state == VADState.SPEAKING and not is_speaking:
                        # 语音结束
                        audio_buffers[session_id].append(audio_chunk)
                        session_states[session_id] = VADState.VOICE_END
                        await websocket.send_json({
                            "type": "vad_event",
                            "event": "voice_end"
                        })
                        info(f"会话 {session_id} 语音结束")

                    elif current_state == VADState.SPEAKING and is_speaking:
                        # 持续说话
                        audio_buffers[session_id].append(audio_chunk)

                    # 可选：从 VOICE_END 回到 IDLE（例如在 reset 或 start 时处理）

                except Exception as e:
                    error(f"处理音频块失败: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"音频处理错误: {str(e)}"
                    })

            elif msg_type == "stop":
                await websocket.send_text(
                    json.dumps({"status": "stopped", "message": "识别结束"})
                )
                break

            elif msg_type == "reset":
                if stream:
                    stream.reset()
                    session_states[session_id] = VADState.IDLE
                    audio_buffers[session_id] = []
                    await websocket.send_json({
                        "type": "info",
                        "message": "VAD 状态已重置"
                    })

            else:
                await websocket.send_json({
                    "type": "warning",
                    "message": "未知消息类型"
                })

    except Exception as e:
        error(f"WebSocket 错误: {e}")
    finally:
        active_streams.pop(session_id, None)
        session_states.pop(session_id, None)
        audio_buffers.pop(session_id, None)
        await websocket.close()
        info("WebSocket 连接已关闭")