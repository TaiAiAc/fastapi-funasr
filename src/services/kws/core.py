# src\services\kws\core.py

from ...utils import debug, info, error
from .streaming import StreamingKWSService
from ...config import config_manager
from funasr import AutoModel
from ..base_model_service import BaseModelService


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

    def create_stream(self):
        """创建流式会话"""
        if not self.is_initialized:
            raise RuntimeError("KWS模型未初始化，请先调用 start()")
        return StreamingKWSService(
            self._model,
        )


# 创建全局KWS服务实例
kws_service = KWSService()

def preload_kws_model() -> bool:
    """预加载 KWS 模型"""
    try:
        kws_service.start()  # 调用基类的 start() 初始化模型
        return kws_service.is_initialized
    except Exception as e:
        error(f"预加载KWS模型失败: {e}")
        return False
