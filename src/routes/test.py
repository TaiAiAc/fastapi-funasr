from fastapi import APIRouter, WebSocket
import json
import numpy as np
from typing import Dict
import uuid

from ..utils import info, error
from ..services import VADStream,vad_service  # 确保这是流式 VAD 服务

websocket_router = APIRouter(
    prefix="/funasr",
    tags=["WebSocket语音识别"],
    responses={404: {"description": "Not found"}},
)

# 管理多个客户端的流
active_streams: Dict[str, VADStream] = {}


@websocket_router.websocket("/ws")
async def websocket_asr(websocket: WebSocket):
    """
    WebSocket语音识别接口
    用于进行语音端点检测(VAD)
    支持检测人声开始和结束节点
    """
    await websocket.accept()
    info("WebSocket 连接已建立")
    
    session_id = str(uuid.uuid4())
    stream = None

    try:
        # 检查VAD服务是否已初始化
        if not vad_service.is_initialized():
            await websocket.send_text(json.dumps({
                "type": "warning", 
                "message": "VAD模型尚未初始化完成，语音端点检测功能将不可用"
            }))
            info("WebSocket连接建立但VAD模型未初始化")
            # 可选择继续或关闭，这里允许继续但不处理音频
        else:
            stream = vad_service.create_stream()
            active_streams[session_id] = stream

        while True:
            # 接收客户端发来的消息（已经是 dict，因为用了 receive_json）
            data = await websocket.receive_json()
            
            msg_type = data.get("type")

            if msg_type == "start":
                await websocket.send_text(json.dumps({"status": "started", "message": "开始接收音频"}))
                info("客户端开始发送音频")

            elif msg_type == "audio":
                if stream is None:
                    # 模型未初始化，跳过处理
                    continue

                try:
                    audio_list = data.get("data")
                    if not isinstance(audio_list, list):
                        raise ValueError("audio data must be a list of floats")

                    # 转为 numpy array (float32, [-1, 1])
                    audio_chunk = np.array(audio_list, dtype=np.float32)

                    # ⚠️ 前端必须保证是 16kHz！这里可加长度校验（可选）
                    # 例如：每帧 160 samples = 10ms @16kHz
                    # if len(audio_chunk) % 160 != 0: ...

                    # 流式处理
                    new_segments = stream.process(audio_chunk)

                    # 如果有新片段，立即返回
                    if new_segments:
                        await websocket.send_json({
                            "type": "vad_result",
                            "segments": new_segments,  # [[start_ms, end_ms], ...]
                            "is_active": stream.is_speech_active()
                        })

                except Exception as e:
                    error(f"处理音频块失败: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"音频处理错误: {str(e)}"
                    })

            elif msg_type == "stop":
                await websocket.send_text(json.dumps({"status": "stopped", "message": "识别结束"}))
                break

            elif msg_type == "reset":
                if stream:
                    stream.reset()
                    await websocket.send_json({"type": "info", "message": "VAD 状态已重置"})

            else:
                await websocket.send_json({"type": "warning", "message": "未知消息类型"})

    except Exception as e:
        error(f"WebSocket 错误: {e}")
    finally:
        active_streams.pop(session_id, None)
        await websocket.close()
        info("WebSocket 连接已关闭")