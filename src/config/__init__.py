"""
配置管理模块，负责统一管理和聚合各种配置来源
"""

from .config_manager import ConfigManager

# 创建全局配置管理器实例（单例模式会确保只创建一次）
config_manager = ConfigManager()

# 导出配置管理器实例，方便其他模块直接使用
def get_config():
    """获取全局配置管理器实例"""
    return config_manager