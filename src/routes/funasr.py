from fastapi import APIRouter
from fastapi import WebSocket
import json
from ..utils.logger import logger, info, error  # 使用统一的日志工具

# 创建WebSocket相关的路由器 - 重命名为更准确的名称
websocket_router = APIRouter(
    prefix="/funasr",  # 路由前缀
    tags=["WebSocket语音识别"],  # 更新标签以更准确反映功能
    responses={404: {"description": "Not found"}},
)

# 修复：使用正确的路由器对象
@websocket_router.websocket("/ws")
async def websocket_asr(websocket: WebSocket):
    await websocket.accept()
    info("WebSocket 连接已建立")  # 使用日志替代print

    try:
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
                    # 这里暂时不处理音频，只回显
                    audio_size = len(message.get("data", ""))
                    info(f"收到音频块，大小: {audio_size} 字节")
                    # 模拟识别结果（后续替换为 FunASR）
                    await websocket.send_text(json.dumps({
                        "type": "result",
                        "text": f"[模拟] 收到 {audio_size} 字节音频"
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