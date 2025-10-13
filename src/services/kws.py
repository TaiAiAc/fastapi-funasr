import os
import time
import torch
from typing import Dict, Any, Optional
from funasr import AutoModel
from ..utils import logger, info, error, debug
from ..config import config_manager

class KWSService:
    """
    关键词检测(KWS)服务类，基于CTC模型实现
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
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化KWS服务，确保模型只被加载一次"""
        if not self._initialized:
            self._initialize_model()

    def _initialize_model(self):
        """
        初始化KWS模型
        使用FunASR官方推荐的AutoModel接口加载CTC KWS模型
        """
        try:
            start_time = time.time()
            info("开始初始化KWS模型...")
            
            # 可以从环境变量或配置文件读取模型参数
            model_name = config_manager.get_kws_config().get('model_name')
            device_config = config_manager.get_kws_config().get('device').lower()

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

            info(f"模型名称: {model_name}")
            info(f"使用设备: {self._device}")

            # 使用官方推荐的AutoModel接口加载模型
            self._model = AutoModel(
                model=model_name,
                device=self._device,
            )

            self._initialized = True
            self._init_time = time.time() - start_time
            info(f"KWS 模型加载成功: {model_name} on {self._device}")
            info(f"KWS模型初始化完成，耗时: {self._init_time:.2f}秒")

        except Exception as e:
            error(f"KWS模型初始化失败: {e}")
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

    def detect_keyword(self, audio_chunk) -> Optional[str]:
        """
        检测音频数据中的关键词
        
        Args:
            audio_chunk: 音频数据，可以是numpy数组或文件路径
        
        Returns:
            str: 检测到的关键词，如果没有检测到关键词则返回None
        """
        try:
            if not self._initialized or self._model is None:
                raise RuntimeError("KWS模型未初始化")

            # 如果是 bytes，尝试解析为 numpy array
            if isinstance(audio_chunk, bytes):
                import io, soundfile as sf
                audio_file = io.BytesIO(audio_chunk)
                data, sr = sf.read(audio_file, dtype='float32')
                if sr != 16000:
                    error("音频采样率非16kHz，KWS结果可能不准确")
                if data.ndim > 1:
                    data = data.mean(axis=1)  # 转单声道
                audio_chunk = data

            start_time = time.time()
            result = self._model.generate(input=audio_chunk)
            process_time = time.time() - start_time
            
            if result and len(result) > 0:
                text = result[0].get("text", "").strip()
                # 官方模型返回 "xiao yun xiao yun" 或 "_silence_"
                detected_keyword = text if text != "_silence_" else None
                if detected_keyword:
                    info(f"检测到关键词: {detected_keyword}，处理时间: {process_time:.4f}秒")
                return detected_keyword
            return None
        except Exception as e:
            error(f"关键词检测失败: {e}")
            return None

# 创建全局KWS服务实例
kws_service = KWSService()

def preload_kws_model() -> bool:
    """
    预加载KWS模型
    
    Returns:
        bool: 预加载是否成功
    """
    try:
        # 强制初始化模型
        kws_service.__init__()
        return kws_service.is_initialized()
    except Exception as e:
        error(f"预加载KWS模型失败: {e}")
        return False