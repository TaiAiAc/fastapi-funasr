from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
from ..utils.logger import logger, info, error  # 使用统一的日志工具

# 创建语音识别相关的路由器
recognition_router = APIRouter(
    prefix="/recognition",  # 路由前缀
    tags=["语音识别"],  # API文档分类标签
    responses={404: {"description": "Not found"}},
)


@recognition_router.post("/asr")
async def speech_recognition(
    audio_file: UploadFile = File(...),
    sample_rate: Optional[int] = Form(16000),
    lang: Optional[str] = Form("zh-cn")
):
    """
    语音识别接口（已移除FunASR功能）
    """
    info(f"语音识别接口被调用，文件名: {audio_file.filename}")
    
    try:
        # 读取音频文件内容
        audio_data = await audio_file.read()
        info(f"读取音频文件完成，大小: {len(audio_data)}字节")
        
        # 返回服务已移除的消息
        return JSONResponse(
            status_code=501,
            content={
                "success": False,
                "error": "FunASR语音识别服务已移除"
            }
        )
    
    except Exception as e:
        error(f"语音识别过程中发生异常: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"服务器内部错误: {str(e)}"}
        )


@recognition_router.post("/stream_asr")
async def stream_speech_recognition(
    audio_chunk: UploadFile = File(...),
    is_final: Optional[bool] = Form(False),
    session_id: Optional[str] = Form(None),
    sample_rate: Optional[int] = Form(16000)
):
    """
    流式语音识别接口（已移除FunASR功能）
    """
    info(f"流式语音识别接口被调用，会话ID: {session_id}, 是否最终块: {is_final}")
    
    try:
        # 返回服务已移除的消息
        return JSONResponse(
            status_code=501,
            content={
                "success": False,
                "error": "FunASR流式语音识别服务已移除"
            }
        )
    
    except Exception as e:
        error(f"流式语音识别过程中发生异常: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"服务器内部错误: {str(e)}"}
        )