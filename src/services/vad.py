import os
import time
from typing import  Dict, Any, List, Tuple
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
            model_name = config_manager.get_vad_config().get('model_name')
            model_revision = config_manager.get_vad_config().get('model_revision')
            device_config = config_manager.get_vad_config().get('device').lower()

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

            # 使用官方推荐的AutoModel接口加载模型
            # 这是FunASR官方推荐且唯一稳定的使用方式
            self._model = AutoModel(
                model=model_name,  # 模型名称（官方支持的）
                model_revision=model_revision,  # 推荐使用最新 revision
                device=self._device,  # 或 "cuda"
                disable_pbar=True,
                disable_log=True,
                # 可选参数
                max_end_silence_time=800,    # 尾部静音超时（ms）
                speech_noise_thres=0.8,      # 语音/噪音阈值（0~1）
            )

            self._initialized = True
            self._init_time = time.time() - start_time
            info(f"流式 VAD 模型加载成功: {model_name}@{model_revision} on {self._device}")
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

    def detect_speech_segments(self, audio_input, **kwargs) -> Dict[str, Any]:
        """
        支持两种输入：
        - str: 音频文件路径
        - np.ndarray: 16kHz 单声道音频数组
        - bytes: WAV 格式音频（需 soundfile 解析）
        """
        try:
            if not self._initialized or self._model is None:
                raise RuntimeError("VAD模型未初始化")

            # 如果是 bytes，尝试解析为 numpy array
            if isinstance(audio_input, bytes):
                import io, soundfile as sf
                audio_file = io.BytesIO(audio_input)
                data, sr = sf.read(audio_file, dtype='float32')
                if sr != 16000:
                    # 可选：重采样（这里简化，建议提前处理）
                    error("音频采样率非16kHz，VAD结果可能不准确")
                if data.ndim > 1:
                    data = data.mean(axis=1)  # 转单声道
                audio_input = data

            start_time = time.time()
            raw_result = self._model.generate(input=audio_input, **kwargs)

            segments = raw_result[0].get("value", []) if raw_result else []
            process_time = time.time() - start_time

            return {
                "segments": segments,  # [[start_ms, end_ms], ...]
                "process_time": process_time,
                "success": True
            }

        except Exception as e:
            error(f"VAD检测失败: {e}")
            return {"segments": [], "error": str(e), "success": False}
      

    def get_audio_segments(self, audio_data: bytes, **kwargs) -> List[Dict[str, Any]]:
        """
        获取音频中的语音片段信息
        
        Args:
            audio_data: 音频数据字节流
            **kwargs: 其他可选参数
                - fs: 采样率，默认为16000
                - max_end_silence_time: 尾部连续静音时间，默认为800ms
                - speech_noise_thres: 语音/噪音阈值，范围(-1, 1)
        
        Returns:
            List[Dict]: 语音片段列表，每个片段包含开始和结束时间
        """
        result = self.detect_speech_segments(audio_data, **kwargs)
        if result.get("success", False):
            return result.get("segments", [])
        return []
    
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
        # 强制初始化模型
        vad_service.__init__()
        return vad_service.is_initialized()
    except Exception as e:
        error(f"预加载VAD模型失败: {e}")
        return False
