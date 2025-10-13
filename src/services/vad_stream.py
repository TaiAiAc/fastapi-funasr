import numpy as np
from typing import List, Dict, Any

class VADStream:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        """重置流状态"""
        self.cache: Dict[str, Any] = {}
        self.total_samples = 0
        self.segments_buffer: List[List[int]] = []

    def process(self, audio_chunk: np.ndarray) -> List[List[int]]:
        """
        处理一段音频块（16kHz, float32, shape=(N,)）
        返回本次新增的语音片段（单位：毫秒）
        """
        if audio_chunk.ndim != 1:
            raise ValueError("audio_chunk 必须是 1D 数组")
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # 调用流式 VAD
        result = self.model.generate(
            input=audio_chunk,
            cache=self.cache,  # 关键：传入并更新状态
            param_dict={
                "vad_streaming": True,        # 启用流式模式
                "return_vad_result": True,    # 返回 VAD 结果而非文本
            }
        )

        new_segments = []
        if result and isinstance(result, list) and len(result) > 0:
            vad_output = result[0]
            if isinstance(vad_output, dict) and "value" in vad_output:
                # value 是 [[start_sample, end_sample], ...]
                for seg in vad_output["value"]:
                    start_ms = int(seg[0] / 16000 * 1000)
                    end_ms = int(seg[1] / 16000 * 1000)
                    new_segments.append([start_ms, end_ms])
                    self.segments_buffer.append([start_ms, end_ms])

        self.total_samples += len(audio_chunk)
        return new_segments

    def is_speech_active(self) -> bool:
        """
        判断当前是否处于“语音活动”状态（即尾部静音未超时）
        在 FunASR 1.2.7 中，可通过 cache 是否包含 'vad_state' 判断
        """
        # 检查 cache 中是否有活跃状态
        vad_cache = self.cache.get("vad_cache", {})
        state = vad_cache.get("vad_state", {})
        # 如果 last_vad_state == 1 表示正在说话或刚结束（未超时）
        return state.get("last_vad_state", 0) == 1
    
    def get_voice_state(self) -> str:
        """
        返回当前语音状态：
        - "silence": 静音
        - "voice_start": 刚开始说话（关键！）
        - "speaking": 正在说话中
        - "voice_end": 刚结束说话
        """
        vad_cache = self.cache.get("vad_cache", {})
        state = vad_cache.get("vad_state", {})
        last_state = state.get("last_vad_state", 0)  # 0=静音, 1=语音
        cur_state = state.get("cur_vad_state", 0)

        # 关键：检测状态跳变
        if last_state == 0 and cur_state == 1:
            return "voice_start"
        elif last_state == 1 and cur_state == 0:
            return "voice_end"
        elif cur_state == 1:
            return "speaking"
        else:
            return "silence"