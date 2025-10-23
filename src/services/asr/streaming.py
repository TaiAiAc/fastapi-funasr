# src/services/asr/streaming.py

from typing import Optional
import numpy as np
from ...utils import info, error
from typing import Dict, Any
from ...utils import debug

class ASRAudioBuffer:
    def __init__(self, target_chunk=9600):
        self.buffer = np.array([], dtype=np.float32)
        self.target_chunk = target_chunk

    def add(self, audio: np.ndarray):
        self.buffer = np.concatenate([self.buffer, audio])

    def get_chunks(self):
        chunks = []
        while len(self.buffer) >= self.target_chunk:
            chunk = self.buffer[: self.target_chunk]
            chunks.append(chunk)
            self.buffer = self.buffer[self.target_chunk :]
        return chunks

class StreamingASRService:
    """
    流式语音识别服务（基于 Paraformer online 模型）
    每个会话应创建独立实例
    """

    def __init__(self, model, generate_options: Dict[str, Any]):
        self._model = model
        self.generate_options = generate_options
        self.reset()

    def reset(self):
        """重置识别状态"""
        self.cache = {}  # 重置cache
        self._partial_text = ""
        self._finalized = False

    def feed_chunk(self, chunk: np.ndarray) -> str:
        # 确保是 float32
        try:
            result = self._model.generate(
                **self.generate_options,
                input=chunk,
                cache=self.cache,
                is_final=False,
            )

            debug(f"模型输出: {result}")

            if isinstance(result, list) and result:
                text = result[0].get("text", "").strip()
                self._partial_text += text  # 覆盖
                return self._partial_text
            else:
                return self._partial_text or ""

        except Exception as e:
            error(f"ASR推理异常: {e}")
            return self._partial_text or ""

    def finalize(self) -> Optional[str]:
        self._finalized = True
        info(f"ASR finalize: 当前部分结果 = '{self._partial_text}'")

        try:
            # 传一个极短的静音（或空）触发 final 输出
            # FunASR 允许 input 为 np.array([]) 或极小数组
            dummy_audio = np.array([], dtype=np.float32)

            result = self._model.generate(
                **self.generate_options,
                input=dummy_audio,
                cache=self.cache,
                is_final=True,
            )

            if isinstance(result, list) and result:
                final_text = result[0].get("text", "").strip()
                # 注意：final_text 可能比 partial 更准确，也可能为空
                # 所以优先用 final_text，若为空则保留 partial
                self._partial_text += final_text

            return self._partial_text

        except Exception as e:
            error(f"ASR finalize 异常: {e}")
        finally:
            self.reset()

    def interrupt(self):
        """打断当前ASR识别流程"""
        self.reset()
