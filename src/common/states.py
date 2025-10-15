# src/common.py
from enum import Enum

class InteractionState(str, Enum):
    IDLE = "idle"
    VAD_ACTIVE = "vad_active"
    KWS_ACTIVE = "kws_active"
    ASR_ACTIVE = "asr_active"
    INTERRUPTING = "interrupting"

class VADState(Enum):
    IDLE = "idle"  # 静音
    SPEAKING = "speaking"  # 正在说话（包含了 voice_start）
    VOICE_END = "voice_end"  # 语音已经结束，等待处理
    PROCESSING = "processing"  # 正在调用ASR、KWS


class KWSState(Enum):
    WAITING = "waiting"  # 等待语音结束
    DETECTING = "detecting"  # 正在分析关键词
    KEYWORD_FOUND = "found"
    NO_KEYWORD = "none"


class ASRState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    DONE = "done"
