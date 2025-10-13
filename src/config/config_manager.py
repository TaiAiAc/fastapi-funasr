import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from threading import Lock

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    配置管理器类，使用严格的单例模式实现
    负责加载、解析和提供配置，确保配置只在服务启动时加载一次
    """
    # 类变量保存单例实例
    _instance = None
    # 线程锁，确保线程安全
    _lock = Lock()
    # 标记配置是否已加载
    _config_loaded = False

    def __new__(cls):
        """创建单例实例的线程安全实现"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化配置管理器，确保配置只加载一次"""
        with self.__class__._lock:
            if not self.__class__._config_loaded:
                # 初始化配置字典
                self._config: Dict[str, Any] = {}
                self._loaded_files: set = set()
                
                # 从配置文件加载配置
                config_file = os.getenv('APP_CONFIG_FILE', 'config.yaml')
                if os.path.exists(config_file):
                    self.load_config_file(config_file)
                
                # 标记配置已加载
                self.__class__._config_loaded = True
                logger.info("配置管理器初始化完成，所有配置已加载")

    def load_config_file(self, file_path: str) -> bool:
        """
        从YAML配置文件加载配置
        
        Args:
            file_path: 配置文件路径
        
        Returns:
            bool: 是否加载成功
        """
        if not os.path.exists(file_path):
            logger.warning(f"配置文件不存在: {file_path}")
            return False
            
        if file_path in self._loaded_files:
            logger.warning(f"配置文件已加载: {file_path}")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                
            if config_data:
                self._merge_config(self._config, config_data)
                self._loaded_files.add(file_path)
                logger.info(f"成功加载配置文件: {file_path}")
                return True
            else:
                logger.warning(f"配置文件为空: {file_path}")
                return False
        except Exception as e:
            logger.error(f"加载配置文件失败: {file_path}, 错误: {str(e)}")
            return False

    def _merge_config(self, base_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """
        合并配置，新配置会覆盖旧配置
        
        Args:
            base_config: 基础配置
            new_config: 新配置
        """
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                self._merge_config(base_config[key], value)
            else:
                # 直接覆盖或添加配置项
                base_config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项，支持点号分隔的嵌套键
        
        Args:
            key: 配置键，如'app.log_level'或'vad.model_name'
            default: 默认值
        
        Returns:
            Any: 配置值或默认值
        """
        parts = key.split('.')
        value = self._config
        
        try:
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        except Exception:
            return default

    def get_app_config(self) -> Dict[str, Any]:
        """获取应用配置"""
        return self._config.get('app', {})

    def get_vad_config(self) -> Dict[str, Any]:
        """获取VAD服务配置"""
        return self._config.get('vad', {})

    def get_kws_config(self) -> Dict[str, Any]:
        """获取KWS服务配置"""
        return self._config.get('kws', {})

    def get_asr_config(self) -> Dict[str, Any]:
        """获取ASR服务配置"""
        return self._config.get('asr', {})

    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._config.copy()

    # 禁用配置更新方法，确保配置只在启动时加载
    def update_config(self, key: str, value: Any) -> bool:
        """更新配置项（禁用，配置应只在启动时加载）"""
        logger.warning("配置更新已禁用，配置应只在服务启动时加载一次")
        return False

    @classmethod
    def reset(cls):
        """
        重置配置管理器（主要用于测试环境）
        生产环境应避免使用此方法
        """
        with cls._lock:
            cls._instance = None
            cls._config_loaded = False
            logger.warning("配置管理器已重置")