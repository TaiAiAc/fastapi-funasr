# src\services\asr\core.py

import os
import time
import torch
from typing import Dict, Any, Optional
from funasr import AutoModel
from ...utils import logger, info, error, debug
from ...config import config_manager


class ASRService:
    """
    自动语音识别(ASR)服务类，基于Paraformer模型实现
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
        """初始化ASR服务，确保模型只被加载一次"""
        if not self._initialized:
            self._initialize_model()

    def _initialize_model(self):
        """
        初始化ASR模型
        使用FunASR官方推荐的AutoModel接口加载Paraformer ASR模型
        """
        try:
            start_time = time.time()
            info("开始初始化ASR模型...")

            # 可以从环境变量或配置文件读取模型参数
            model_name = config_manager.get_asr_config().get("model_name")
            model_revision = config_manager.get_asr_config().get("model_revision")
            device_config = config_manager.get_asr_config().get("device").lower()

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

            info(f"正在加载ASR模型，使用官方AutoModel接口")
            info(f"模型名称: {model_name}")
            info(f"模型版本: {model_revision}")
            info(f"使用设备: {self._device}")

            # 使用官方推荐的AutoModel接口加载模型
            self._model = AutoModel(
                model=model_name,
                model_revision=model_revision,
                device=self._device,
                disable_pbar=True,
                disable_log=True,
                # paraformer 支持流式
                chunk_size=5,  # 可选
            )

            self._initialized = True
            self._init_time = time.time() - start_time
            info(f"ASR 模型加载成功: {model_name}@{model_revision} on {self._device}")
            info(f"ASR模型初始化完成，耗时: {self._init_time:.2f}秒")

        except Exception as e:
            error(f"ASR模型初始化失败: {e}")
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

    def transcribe(self, audio_data) -> str:
        """
        将音频数据转换为文本

        Args:
            audio_data: 音频数据，可以是numpy数组或文件路径

        Returns:
            str: 转换后的文本结果，如果转换失败则返回空字符串
        """
        try:
            if not self._initialized or self._model is None:
                raise RuntimeError("ASR模型未初始化")

            # 如果是 bytes，尝试解析为 numpy array
            if isinstance(audio_data, bytes):
                import io, soundfile as sf

                audio_file = io.BytesIO(audio_data)
                data, sr = sf.read(audio_file, dtype="float32")
                if sr != 16000:
                    error("音频采样率非16kHz，ASR结果可能不准确")
                if data.ndim > 1:
                    data = data.mean(axis=1)  # 转单声道
                audio_data = data

            start_time = time.time()
            result = self._model.generate(input=audio_data)
            process_time = time.time() - start_time

            if result and len(result) > 0:
                text = result[0].get("text", "").strip()
                info(f"ASR识别结果: {text}，处理时间: {process_time:.4f}秒")
                return text
            return ""
        except Exception as e:
            error(f"语音识别失败: {e}")
            return ""


# 创建获取ASR服务实例的函数
def get_asr_service() -> ASRService:
    """
    获取ASR服务实例（懒加载模式）
    只有在首次调用此函数时才会初始化模型，避免模块导入时的资源消耗

    Returns:
        ASRService: 单例的ASR服务实例
    """
    # 检查是否已初始化
    if not hasattr(get_asr_service, "_instance"):
        # 首次调用时创建实例
        get_asr_service._instance = ASRService()
    return get_asr_service._instance


def preload_asr_model() -> bool:
    """
    预加载ASR模型

    Returns:
        bool: 预加载是否成功
    """
    try:
        # 使用新的get_asr_service函数获取实例
        service = get_asr_service()
        # 强制初始化模型
        if not service.is_initialized():
            service._initialize_model()
        return service.is_initialized()
    except Exception as e:
        error(f"预加载ASR模型失败: {e}")
        return False
