import json
import uuid
import numpy as np
from typing import Dict
from fastapi import APIRouter, WebSocket

from ..common import VADState
from ..utils import info, error
from ..services.vad import vad_service  # 确保这是流式 VAD 服务
from ..services.vad_stream import VADStream  # 确保这是流式 VAD 服务

# 创建WebSocket相关的路由器
websocket_router = APIRouter(
    prefix="/funasr",  # 路由前缀
    tags=["WebSocket语音识别"],  # API文档标签
    responses={404: {"description": "Not found"}},
)

# 管理多个客户端的流
active_streams: Dict[str, VADStream] = {}

# 每个会话的状态
session_states: Dict[str, VADState] = {}
audio_buffers: Dict[str, list] = {}  # 缓存音频用于 ASR


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
    prev_vad_state = VADState.IDLE

    info(f"WebSocket 连接已建立，会话ID: {session_id}")

    try:
        # 检查VAD服务是否已初始化
        if not vad_service.is_initialized():
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "warning",
                        "message": "VAD模型尚未初始化完成，语音端点检测功能将不可用",
                    }
                )
            )
            info("WebSocket连接建立但VAD模型未初始化")
        else:
            stream = vad_service.create_stream()
            active_streams[session_id] = stream
            audio_buffers[session_id] = []

        while True:
            # 接收客户端发来的消息（可以是音频数据或控制命令）
            data = await websocket.receive_json()
            # 修复：使用data变量而不是未定义的message变量
            msg_type = data.get("type")

            if msg_type == "start":
                session_states[session_id] = VADState.IDLE
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
                    print("接受音频数据:", audio_list[:10])  # 打印前10个样本
                    if not isinstance(audio_list, list):
                        raise ValueError("audio data must be a list of floats")

                    # 转为 numpy array (float32, [-1, 1])
                    audio_chunk = np.array(audio_list, dtype=np.float32)
                    # ✅ 关键：验证数值范围
                    if not (-1.0 <= audio_chunk.min() <= 1.0 and -1.0 <= audio_chunk.max() <= 1.0):
                        error(f"音频超出 [-1,1] 范围！")
                        await websocket.send_json({"type": "error", "message": "音频数据格式错误"})
                        continue
                    
                    # ✅ 添加音频块长度检查
                    if len(audio_chunk) % 160 != 0:  # 160 = 16000 / 1000 * 10ms
                        error(f"音频块长度 {len(audio_chunk)} 不是160的倍数！")
                        await websocket.send_json({"type": "error", "message": "音频帧长度错误"})
                        continue
                    
                    print("前10个样本:", audio_chunk[:10])

                    # 流式处理
                    stream = active_streams[session_id]
                    stream.process(audio_chunk)

                    # 获取当前 VAD 状态
                    current_vad_state = stream.get_voice_state()
                    info(f"Current VAD state: {current_vad_state}")

                    # ✅ 修复状态机逻辑：允许从IDLE或VOICE_END状态转换到SPEAKING
                    if (
                        (prev_vad_state == VADState.IDLE or prev_vad_state == VADState.VOICE_END)
                        and current_vad_state == VADState.SPEAKING
                    ):
                        # 语音开始：清空缓冲区 + 加入当前音频
                        audio_buffers[session_id] = [audio_chunk]  # 关键：清空并初始化
                        session_states[session_id] = VADState.SPEAKING
                        await websocket.send_json(
                            {
                                "type": "vad_event",
                                "event": "voice_start",
                                "message": "检测到人声开始",
                            }
                        )
                        info(f"会话 {session_id} 开始说话")

                    elif (
                        prev_vad_state == VADState.SPEAKING
                        and current_vad_state == VADState.VOICE_END
                    ):
                        # voice_end 事件
                        audio_buffers[session_id].append(audio_chunk)  # 加入最后一块
                        await websocket.send_json(
                            {"type": "vad_event", "event": "voice_end"}
                        )
                        session_states[session_id] = VADState.VOICE_END
                        info(f"会话 {session_id} 语音结束")

                    elif current_vad_state == VADState.SPEAKING:
                        # 持续说话，继续缓存
                        audio_buffers[session_id].append(audio_chunk)
                        session_states[session_id] = VADState.SPEAKING
                        info(f"会话 {session_id} 正在说话")

                    elif (
                        current_vad_state == VADState.IDLE
                        and prev_vad_state == VADState.VOICE_END
                    ):
                        # 回到静音
                        session_states[session_id] = VADState.IDLE

                    prev_vad_state = current_vad_state

                except Exception as e:
                    error(f"处理音频块失败: {e}")
                    await websocket.send_json(
                        {"type": "error", "message": f"音频处理错误: {str(e)}"}
                    )

            elif msg_type == "stop":
                await websocket.send_text(
                    json.dumps({"status": "stopped", "message": "识别结束"})
                )
                break

            elif msg_type == "reset":
                if stream:
                    stream.reset()
                    await websocket.send_json(
                        {"type": "info", "message": "VAD 状态已重置"}
                    )

            else:
                await websocket.send_json(
                    {"type": "warning", "message": "未知消息类型"}
                )

    except Exception as e:
        error(f"WebSocket 错误: {e}")
    finally:
        active_streams.pop(session_id, None)
        await websocket.close()
        info("WebSocket 连接已关闭")
