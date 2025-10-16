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
        if not self.is_initialized:
            raise RuntimeError("KWS 模型未初始化")

        # ✅ 确保输入是 float32 且在 [-1.0, 1.0]
        if audio.dtype == np.int16:
            # 如果意外收到 int16，转为 float32
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.float32:
            # 已是 float32，只需 clip
            pass
        else:
            raise ValueError(f"不支持的音频 dtype: {audio.dtype}")

        audio = np.clip(audio, -1.0, 1.0)

        try:
            result = self._model.generate(input=audio)  # ← 传 float32！
            if not isinstance(result, list) or len(result) == 0:
                return None

            res = result[0]
            if not isinstance(res, dict):
                return None

            raw_text = res.get("text", "")
            if not isinstance(raw_text, str) or not raw_text.startswith("detected "):
                return None

            try:
                parts = raw_text.split()
                if len(parts) < 3:
                    return None
                keyword = parts[1]
                score = float(parts[2])
            except (ValueError, IndexError):
                error(f"无法解析 KWS 输出: {raw_text}")
                return None

            keyword = keyword.strip()
            info(f"检测到关键词: {keyword}，置信度: {score:.4f}")
            if not keyword or score < 0.2:
                return None
            return keyword

        except Exception as e:
            error(f"KWS detect_keyword 异常: {e}")
            return None

    def create_stream(self):
        """创建流式会话"""
        if not self.is_initialized:
            raise RuntimeError("KWS模型未初始化，请先调用 start()")
        return StreamingKWSService(self)
