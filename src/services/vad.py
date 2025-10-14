import os
import time
from typing import Dict, Any, List, Tuple
from ..utils import logger, info, error, debug
from typing import List
from .vad_stream import VADStream
from ..config import config_manager
import torch
from funasr import AutoModel


class VADService:
    """
    语音端点检测(VAD)服务类，基于FSMN模型实现
    使用单例模式确保模型只被加载一次
    """

    _instance = None
    _model = None
    _initialized = False
    _init_time = 0.0
    _device = None

    def __new__(cls):
        """创建单例实例"""
        if cls._instance is None:
            cls._instance = super(VADService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化VAD服务，确保模型只被加载一次"""
        if not self._initialized:
            self._initialize_model()

    def _initialize_model(self):
        """
        初始化VAD模型
        使用FunASR官方推荐的AutoModel接口加载FSMN VAD模型
        """
        try:
            start_time = time.time()
            info("开始初始化VAD模型...")

            # 可以从环境变量或配置文件读取模型参数
            model_name = config_manager.get_vad_config().get("model_name")
            model_revision = config_manager.get_vad_config().get("model_revision")
            model_params = config_manager.get_vad_config().get("params")
            device_config = config_manager.get_vad_config().get("device").lower()

            if device_config == "auto":
                # 自动检测GPU可用性
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            elif device_config in ["cuda", "cpu"]:
                # 使用指定的设备
                self._device = device_config
            else:
                # 配置无效，使用默认逻辑
                error(f"无效的设备配置: {device_config}，将使用自动检测模式")
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            info(f"正在加载VAD模型，使用官方AutoModel接口")
            info(f"模型名称: {model_name}")
            info(f"模型版本: {model_revision}")
            info(f"使用设备: {self._device}")
            info(f"其他参数: {model_params}")

            # 使用官方推荐的AutoModel接口加载模型
            # 这是FunASR官方推荐且唯一稳定的使用方式
            self._model = AutoModel(
                model=model_name,  # 模型名称（官方支持的）
                model_revision=model_revision,  # 推荐使用最新 revision
                # device=self._device,  # 或 "cuda"
                # 可选参数
                # max_end_silence_time=300,  # ✅ 从800ms → 300ms（更敏感）
                # speech_noise_thres=0.15,  # 保持合理阈值
            )

            self._initialized = True
            self._init_time = time.time() - start_time
            info(
                f"流式 VAD 模型加载成功: {model_name}@{model_revision} on {self._device}"
            )
            info(f"VAD模型初始化完成，耗时: {self._init_time:.2f}秒")

        except Exception as e:
            error(f"VAD模型初始化失败: {e}")
            self._initialized = False
            raise

    def is_initialized(self) -> bool:
        """检查模型是否已初始化完成"""
        return self._initialized

    def get_init_info(self) -> Dict[str, Any]:
        """获取初始化信息"""
        return {
            "initialized": self._initialized,
            "init_time": self._init_time,
            "model_available": self._model is not None,
            "device": self._device,
        }

    def create_stream(self):
        """创建一个新的流式会话（每个说话人/会话一个）"""
        if not self._initialized:
            raise RuntimeError("VAD模型未初始化")
        return VADStream(self._model)


# 创建全局VAD服务实例
vad_service = VADService()

def preload_vad_model() -> bool:
    """
    预加载VAD模型

    Returns:
        bool: 预加载是否成功
    """
    try:
        # ✅ 添加设备检查
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 确保使用GPU
        # 强制初始化模型
        vad_service.__init__()
        return vad_service.is_initialized()
    except Exception as e:
        error(f"预加载VAD模型失败: {e}")
        return False
