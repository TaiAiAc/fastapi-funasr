# src\services\kws\core.py

from ...utils import debug, info, error
from .streaming import StreamingASRService
from ...common import global_config
from funasr import AutoModel
from ..base_model_service import BaseModelService


class ASRService(BaseModelService):
    """
    语音识别(ASR)服务类，基于Paraformer模型实现
    使用单例模式确保模型只被加载一次
    提供模型管理和流式会话创建功能
    """

    def __init__(self):
        """初始化ASR服务，确保模型只被加载一次"""
        # 从配置读取
        self.asr_config = global_config.get_asr_config()

        super().__init__(service_name="ASR服务", model_options=self.asr_config)

    def _load_model(self, **kwargs):
        """加载ASR模型"""
        return AutoModel(**kwargs)

    def infer(self, audio_input, **kwargs):
        """非流式推理"""
        return self._model.generate(input=audio_input, **kwargs)

    def create_stream(self):
        """创建流式会话"""
        if not self.is_initialized:
            raise RuntimeError("ASR模型未初始化，请先调用 start()")
        return StreamingASRService(self._model, self.asr_config.get("generate", {}))
