# src\services\kws\core.py

from ...utils import debug, info, error
from .streaming import StreamingKWSService
from ...config import config_manager
from funasr import AutoModel
from ..base_model_service import BaseModelService
import numpy as np


class KWSService(BaseModelService):
    """
    关键词 spotting(KWS)服务类，基于FSMN模型实现
    使用单例模式确保模型只被加载一次
    提供模型管理和流式会话创建功能
    """

    def __init__(self):
        """初始化KWS服务，确保模型只被加载一次"""
        # 从配置读取
        kws_config = config_manager.get_kws_config()

        super().__init__(
            model_name=kws_config.get("model_name"),
            device=kws_config.get("device", "auto"),
            init_params=kws_config.get("params", {}),
            service_name="KWS服务",
        )

    def _load_model(self, **kwargs):
        """加载KWS模型"""
        return AutoModel(**kwargs)

    def infer(self, audio_input, **kwargs):
        """非流式推理"""
        return self._model.generate(input=audio_input, **kwargs)

    def detect_keyword(self, audio: np.ndarray) -> str | None:
        """
        对一段音频进行关键词检测
        :param audio: int16 或 float32, 16k, 单通道
        :return: 唤醒词文本（如 "小云"），未命中返回 None
        """
        if not self.is_initialized:
            raise RuntimeError("KWS 模型未初始化")

        # FunASR 要求输入为 int16
        if audio.dtype == np.float32:
            audio = (np.clip(audio, -1.0, 1.0) * 32768).astype(np.int16)

        try:
            result = self._model.generate(input=audio)
            # FunASR KWS 返回格式：[{'text': '小云', 'score': 0.92}]
            if isinstance(result, list) and len(result) > 0:
                res = result[0]
                if isinstance(res, dict) and res.get("text"):
                    keyword = res["text"].strip()
                    score = res.get("score", 0.0)
                    if score >= 0.5:  # 可配置阈值
                        return keyword
            return None
        except Exception as e:
            error(f"KWS detect_keyword 异常: {e}")
            return None

    def create_stream(self):
        """创建流式会话"""
        if not self.is_initialized:
            raise RuntimeError("KWS模型未初始化，请先调用 start()")
        return StreamingKWSService(
            self._model,
        )
