from fastapi import APIRouter
from ..utils.logger import logger  # 使用统一的日志工具

# 创建语音识别相关的路由器
recognition_router = APIRouter(
    prefix="/recognition",  # 路由前缀
    tags=["语音识别"],  # 用于API文档分类
    responses={404: {"description": "Not found"}},
)


@recognition_router.post("/asr")
async def speech_recognition():
    """
    语音识别接口
    用于将音频文件转换为文本
    """
    logger.info("语音识别接口被调用")
    # 实际实现代码
    return {"text": "识别结果"}


@recognition_router.post("/stream_asr")
async def stream_speech_recognition():
    """
    流式语音识别接口
    支持实时音频流的识别
    """
    logger.info("流式语音识别接口被调用")
    # 实际实现代码
    return {"text": "流式识别结果"}