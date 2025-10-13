import os
import time
from typing import Optional, Dict, Any, List, Tuple
from ..utils.logger import logger, info, error, debug

# 尝试加载.env文件中的环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()  # 加载.env文件中的环境变量
except ImportError:
    debug("dotenv模块未安装，无法加载.env文件中的环境变量")

# 导入FunASR相关库
try:
    import funasr
    from funasr import AutoModel
    import torch
    import os
except ImportError as e:
    error(f"导入FunASR相关库失败: {e}")
    raise


class VADService:
    """
    语音端点检测(VAD)服务类，基于FSMN模型实现
    使用单例模式确保模型只被加载一次
    """
    _instance = None
    _model = None
    _initialized = False
    _init_time = 0.0
    
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
            model_name = os.getenv("FUNASR_VAD_MODEL_NAME", "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")
            model_revision = os.getenv("FUNASR_VAD_MODEL_REVISION", "v2.0.4")
                
            # 读取设备配置
            device_config = os.getenv("FUNASR_VAD_DEVICE", "auto").lower()
            
            if device_config == "auto":
                # 自动检测GPU可用性
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif device_config in ["cuda", "cpu"]:
                # 使用指定的设备
                device = device_config
            else:
                # 配置无效，使用默认逻辑
                error(f"无效的设备配置: {device_config}，将使用自动检测模式")
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            info(f"正在加载VAD模型，使用官方AutoModel接口")
            info(f"模型名称: {model_name}")
            info(f"模型版本: {model_revision}")
            info(f"使用设备: {device}")
            
            # 使用官方推荐的AutoModel接口加载模型
            # 这是FunASR官方推荐且唯一稳定的使用方式
            self._model = AutoModel(
                model=model_name,                # 模型名称（官方支持的）
                model_revision=model_revision,    # 推荐使用最新 revision
                disable_pbar=True,
                device=device                    # 或 "cuda"
            )
            
            self._initialized = True
            self._init_time = time.time() - start_time
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
            "model_available": self._model is not None
        }
    
    def detect_speech_segments(self, audio_data: bytes, **kwargs) -> Dict[str, Any]:
        """
        检测音频中的语音片段
        
        Args:
            audio_data: 音频数据字节流
            **kwargs: 其他可选参数
                - fs: 采样率，默认为16000
                - max_end_silence_time: 尾部连续静音时间，默认为800ms
                - speech_noise_thres: 语音/噪音阈值，范围(-1, 1)
        
        Returns:
            Dict: 包含语音片段信息的结果
        """
        try:
            if not self._initialized or self._model is None:
                error("VAD模型未初始化，无法执行检测")
                raise RuntimeError("VAD模型未初始化")
            
            start_time = time.time()
            debug(f"开始处理音频数据，大小: {len(audio_data)}字节")
            
            # 执行语音端点检测
            result = self._model.generate(
                input=audio_data,
                **kwargs
            )
            
            process_time = time.time() - start_time
            debug(f"VAD处理完成，耗时: {process_time:.2f}秒")
            
            # 格式化检测结果
            formatted_result = {
                "segments": result.get("value", []),
                "process_time": process_time,
                "success": True
            }
            
            return formatted_result
            
        except Exception as e:
            error(f"语音端点检测过程中发生错误: {e}")
            return {
                "segments": [],
                "error": str(e),
                "success": False
            }
    
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