# src/routes/funasr.py

import json
import uuid
import numpy as np
from typing import Dict, Optional
from fastapi import APIRouter, WebSocket

from ..utils import debug,info, error
from ..services import vad_service, VADStateMachine, get_kws_service, get_asr_service
from ..utils.audio_converter import AudioConverter

kws_service = get_kws_service()
asr_service = get_asr_service()

websocket_router = APIRouter(
    prefix="/funasr",
    tags=["WebSocket语音识别"],
    responses={404: {"description": "Not found"}},
)

# 客户端状态管理
active_state_machines: Dict[str, VADStateMachine] = {}


async def on_vad_start(websocket: WebSocket, session_id: str):
    await websocket.send_json(
        {
            "type": "vad_event",
            "event": "voice_start",
            "message": f"会话 {session_id} 检测到人声开始",
        }
    )


async def on_kws_feed(websocket: WebSocket, session_id: str, chunk: np.ndarray):
    """流式送入 KWS 模型"""
    if not kws_service.is_initialized():
        return
    pass


async def on_asr_feed(websocket: WebSocket, session_id: str, chunk: np.ndarray):
    """流式送入 ASR 模型"""
    if not asr_service.is_initialized():
        return
    partial_result = await asr_service.recognize_stream(chunk)
    if partial_result:
        await websocket.send_json({"type": "asr_partial", "text": partial_result})


async def on_asr_end(websocket: WebSocket, session_id: str, final_audio: np.ndarray):
    """ASR 结束，可做最终识别（可选）"""
    if asr_service.is_initialized():
        pass
        await websocket.send_json({"type": "asr_final", "text": ""})
    # 重置状态机
    sm = active_state_machines.get(session_id)
    if sm:
        sm.reset()
    await websocket.send_json(
        {"type": "vad_event", "event": "asr_end", "message": "语音识别结束"}
    )


async def on_interrupt(websocket: WebSocket, session_id: str):
    """被打断：停止当前 ASR，准备新识别"""
    await asr_service.interrupt()
    await websocket.send_json(
        {"type": "interrupt", "message": "检测到唤醒词，已打断当前识别"}
    )


def bind_callbacks(sm: VADStateMachine, websocket: WebSocket, session_id: str):
    """绑定状态机回调"""
    sm.on_vad_start = lambda: on_vad_start(websocket, session_id)
    sm.on_kws_feed = lambda chunk: on_kws_feed(websocket, session_id, chunk)
    sm.on_asr_feed = lambda chunk: on_asr_feed(websocket, session_id, chunk)
    sm.on_asr_end = lambda audio: on_asr_end(websocket, session_id, audio)
    sm.on_interrupt = lambda: on_interrupt(websocket, session_id)


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
        else:
            stream = vad_service.create_stream(max_end_silence_time=600,speech_noise_thres=0.8)

        # 创建状态机
        sm = VADStateMachine(
            sample_rate=16000,
            min_speech_ms=200,
            max_speech_ms=15000,
            trailing_silence_ms=600,
            buffer_duration_sec=5.0,
        )
        bind_callbacks(sm, websocket, session_id)
        active_state_machines[session_id] = sm

        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "start":
                sm.reset()
                if stream:
                    stream.reset()
                await websocket.send_json(
                    {"status": "started", "message": "开始接收音频"}
                )
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

                    # === 核心：更新状态机 ===
                    sm.add_audio_chunk(audio_chunk)  # 缓存 float32

                    # 转为 int16 供 VAD 使用
                    audio_int16 = AudioConverter.to_int16(audio_chunk)

                    rms = np.sqrt(np.mean(audio_chunk**2))
                    debug(f"Audio chunk RMS: {rms:.6f}, len: {len(audio_chunk)}")
                    debug(f"Audio chunk min: {audio_chunk.min():.6f}, max: {audio_chunk.max():.6f}")    
                    debug(f"Audio chunk int16 min: {audio_int16.min():.6f}, max: {audio_int16.max():.6f}")

                    vad_segments = stream.process(audio_int16)
                    if vad_segments:
                        await sm.on_vad_segments(vad_segments)

                except Exception as e:
                    error(f"处理音频失败: {e}")
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif msg_type == "stop" or msg_type == "reset":
                sm.reset()
                if stream:
                    stream.reset()
                await websocket.send_json({"type": "info", "message": "已重置"})
                if msg_type == "stop":
                    break

            else:
                await websocket.send_json(
                    {"type": "warning", "message": "未知消息类型"}
                )

    except Exception as e:
        error(f"WebSocket 错误: {e}")
    finally:
        active_state_machines.pop(session_id, None)
        # 检查WebSocket连接是否仍然打开，避免重复关闭
        if websocket.client_state == 1:  # 1表示连接仍然打开
            await websocket.close()
        info("WebSocket 连接已关闭")
