# src/services/asr/streaming.py

from typing import Optional
import numpy as np
from ...utils import info, error
from typing import Dict, Any


class StreamingASRService:
    """
    流式语音识别服务（基于 Paraformer online 模型）
    每个会话应创建独立实例
    """

    def __init__(self, generate_options: Dict[str, Any]):
        self.generate_options = generate_options
        self.reset()

    def reset(self):
        """重置识别状态"""
        self.cache = {}  # 重置cache
        self._total_samples = 0
        self._partial_text = ""
        self._finalized = False

    def feed_chunk(self, chunk: np.ndarray) -> Optional[str]:
        """
        输入音频块，返回部分识别结果（partial）或最终结果（final）
        """

        # 转为int16格式（根据模型要求）
        if chunk.dtype == np.float32:
            chunk = (chunk * 32768.0).astype(np.int16)

        # 计算音频块时长（毫秒）
        chunk_ms = len(chunk) / 16.0

        # 流式推理
        try:
            result = self._model.generate(
                **self.generate_options,
                input=chunk,
                cache=self.cache,  # 传递cache保持模型状态
                is_final=False,  # 非最终帧
                return_cache=True,  # 返回更新后的cache
            )

            # 解析结果
            partial_text = result[0].get("text", "").strip()
            self._partial_text += partial_text

            # 返回中间结果
            return partial_text

        except Exception as e:
            error(f"ASR流式推理异常: {e}")
            return None

    def finalize(self) -> Optional[str]:
        """强制结束识别，返回最终结果"""
        # 最终帧处理
        if self._finalized:
            return self._partial_text

        try:
            # 设置is_final=True触发最终结果输出
            result = self._model.generate(
                input=np.array([], dtype=np.int16),  # 传递空音频
                cache=self.cache,
                is_final=True,
                chunk_size=[0, 10, 5],
                return_cache=True,
            )

            final_text = result[0].get("text", "").strip()
            self._partial_text += final_text

            self._finalized = True
            self.reset()  # 处理完成后重置状态
            return self._partial_text

        except Exception as e:
            error(f"ASR最终结果处理异常: {e}")
            return None

    async def interrupt(self):
        """打断当前ASR识别流程"""
        if self._asr_stream:
            self._asr_stream.reset()
        self._current_state = VADState.IDLE
        self._partial_text = ""
        self._finalized = False
