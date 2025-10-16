# src/services/base_model_service.py

import time
import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
from ..utils import info, error, warning, resolve_device


class BaseModelService(ABC):
    """
    通用模型服务基类，支持流式与非流式推理
    所有具体服务（VAD/KWS/ASR）应继承此类
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        init_params: Optional[Dict[str, Any]] = None,
        service_name: str = "模型服务",
    ):
        self.model_name = model_name
        self.device_config = device
        self.init_params = init_params or {}
        self.service_name = service_name

        self._model = None
        self._device = None
        self._is_initialized = False
        self._is_streaming = False

    def start(self):
        """启动服务：加载模型"""
        if self._is_initialized:
            warning(f"{self.service_name} 已启动，跳过重复初始化")
            return

        start_time = time.time()
        info(f"🚀 启动 {self.service_name}...")

        # 解析设备
        self._device = resolve_device(self.device_config)

        # 构建模型参数
        model_params = {
            "model": self.model_name,
            "device": self._device,
            **self.init_params,
        }

        try:
            info(f"正在加载 {self.service_name} 模型参数: {self.init_params}")

            # 调用子类实现的 _load_model 方法
            self._model = self._load_model(**model_params)

            self._is_initialized = True
            elapsed = time.time() - start_time
            info(f"✅ {self.service_name} 启动成功，耗时: {elapsed:.2f} 秒")
        except Exception as e:
            error(f"❌ {self.service_name} 启动失败: {e}")
            raise

    def stop(self):
        """停止服务：释放资源（可选）"""
        if self._model is not None:
            del self._model
            torch.cuda.empty_cache()
            info(f"⏹️  {self.service_name} 已停止")
        self._is_initialized = False
        self._is_streaming = False

    def get_init_info(self) -> Dict[str, Any]:
        """获取初始化信息"""
        return {
            "model_name": self.model_name,
            "initialized": self._is_initialized,
            "init_time": self._init_time,
            "model_available": self._model is not None,
            "device": self._device,
        }

    @abstractmethod
    def _load_model(self, **kwargs) -> Any:
        """
        子类必须实现：如何加载模型（如 AutoModel(...)）
        """
        pass

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming
