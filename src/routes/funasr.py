from fastapi import APIRouter
from fastapi import WebSocket
import json
import base64
from ..utils.logger import logger, info, error, debug  # 使用统一的日志工具
from ..services import vad_service  # 导入VAD服务

# 创建WebSocket相关的路由器
websocket_router = APIRouter(
    prefix="/funasr",  # 路由前缀
    tags=["WebSocket语音识别"],  # API文档标签
    responses={404: {"description": "Not found"}},
)


@websocket_router.websocket("/ws")
async def websocket_asr(websocket: WebSocket):
    """
    WebSocket语音识别接口
    用于进行语音端点检测(VAD)
    支持检测人声开始和结束节点
    """
    await websocket.accept()
    info("WebSocket 连接已建立")

    try:
        # 检查VAD服务是否已初始化
        if not vad_service.is_initialized():
            await websocket.send_text(json.dumps({
                "type": "warning", 
                "message": "VAD模型尚未初始化完成，语音端点检测功能将不可用"
            }))
            info("WebSocket连接建立但VAD模型未初始化")
        
        # 跟踪上一帧的语音状态
        last_voice_state = False
        # 保存当前的语音段
        current_utterance = b""

        while True:
            # 接收客户端发来的消息（可以是音频数据或控制命令）
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                msg_type = message.get("type")

                if msg_type == "start":
                    await websocket.send_text(json.dumps({"status": "started", "message": "开始接收音频"}))
                    info("客户端开始发送音频")

                elif msg_type == "audio":
                    # 获取音频数据并进行Base64解码
                    audio_base64 = message.get("data", "")
                    try:
                        audio_data = base64.b64decode(audio_base64)
                        audio_size = len(audio_data)
                        info(f"收到音频块，大小: {audio_size} 字节")

                        # 保存当前音频数据用于VAD检测
                        current_utterance += audio_data
                        
                        # 如果VAD服务已初始化，进行语音端点检测
                        if vad_service.is_initialized():
                            # 执行VAD检测
                            vad_result = vad_service.detect_speech_segments(
                                audio_data=audio_data,
                                fs=message.get("sample_rate", 16000),
                                max_end_silence_time=800,  # 800ms的尾部静音判定为语音结束
                                speech_noise_thres=0.5  # 语音/噪音阈值
                            )
                            
                            # 检查是否检测到语音
                            current_voice_state = len(vad_result.get("segments", [])) > 0
                            
                            # 检测到语音开始
                            if current_voice_state and not last_voice_state:
                                await websocket.send_text(json.dumps({
                                    "type": "vad_start",
                                    "message": "检测到说话开始"
                                }))
                                debug("检测到说话开始")
                            
                            # 检测到语音结束
                            elif not current_voice_state and last_voice_state:
                                # 发送语音结束通知
                                await websocket.send_text(json.dumps({
                                    "type": "vad_end",
                                    "message": "检测到说话结束"
                                }))
                                debug("检测到说话结束")
                                
                                # 重置当前语音段
                                current_utterance = b""
                            
                            # 更新语音状态
                            last_voice_state = current_voice_state
                        
                        # 由于移除了FunASR服务，这里只返回VAD结果
                        await websocket.send_text(json.dumps({
                            "type": "result",
                            "text": "FunASR服务已移除",
                            "has_voice": last_voice_state if vad_service.is_initialized() else None
                        }))

                    except Exception as e:
                        error(f"处理音频数据时出错: {e}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": f"处理音频数据时出错: {str(e)}"
                        }))

                elif msg_type == "stop":
                    await websocket.send_text(json.dumps({"status": "stopped", "message": "识别结束"}))
                    break

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "无效的 JSON 格式"}))
                error("收到无效的JSON格式数据")

    except Exception as e:
        error(f"WebSocket 错误: {e}")
    finally:
        await websocket.close()
        info("WebSocket 连接已关闭")