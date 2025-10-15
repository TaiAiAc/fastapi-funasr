# src/common.py
from enum import Enum

class InteractionState(str, Enum):
    IDLE = "idle"
    VAD_ACTIVE = "vad_active"      # VAD 检测到语音，正在收集中（KWS 流式监听）
    KWS_ACTIVE = "kws_active"      # 【可选】KWS 正在分析（实际可合并到 VAD_ACTIVE）
    ASR_ACTIVE = "asr_active"      # ASR 正在识别
    INTERRUPTING = "interrupting"  # 检测到新唤醒词，打断当前 ASR

# 保留 VADState 用于 VADSession 内部（可选）
class VADState(Enum):
    IDLE = "idle"
    SPEAKING = "speaking"
    VOICE_END = "voice_end"

# KWSState / ASRState 可保留，但建议服务内部使用，不暴露给状态机
class KWSState(Enum):
    WAITING = "waiting"
    DETECTING = "detecting"
    KEYWORD_FOUND = "found"
    NO_KEYWORD = "none"

class ASRState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    DONE = "done"