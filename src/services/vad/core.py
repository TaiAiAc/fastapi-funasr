# src\services\vad\core.py

from .streaming import StreamingVADService
from ...common import global_config
from funasr import AutoModel
from ..base_model_service import BaseModelService


class VADService(BaseModelService):
    """
    语音端点检测(Voice Activity Detection, VAD)服务类，基于FSMN模型实现
    使用单例模式确保模型只被加载一次
    提供模型管理和流式会话创建功能
    """

    def __init__(self):
        """初始化VAD服务，确保模型只被加载一次"""
        # 从配置读取
        self.vad_config = global_config.get_vad_config()

        super().__init__(service_name="VAD服务", model_options=self.vad_config)

    def _load_model(self, **kwargs):
        """加载VAD模型"""
        return AutoModel(**kwargs)

    def infer(self, audio_input, **kwargs):
        """非流式推理"""
        return self._model.generate(input=audio_input, **kwargs)

    def create_stream(
        self,
        merge_gap_ms: int = 50,
    ):
        """创建流式会话"""
        if not self.is_initialized:
            raise RuntimeError("VAD模型未初始化，请先调用 start()")
        return StreamingVADService(
            self._model,
            self.vad_config.get("generate", {}),
            merge_gap_ms=merge_gap_ms,
        )
