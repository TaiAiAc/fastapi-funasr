import logging
import os
import datetime
from ..config import config_manager

class Logger:
    """
    统一日志工具类，提供简单统一的日志功能
    """
    _instance = None
    _logger = None
    _initialized = False

    @classmethod
    def get_logger(cls, name: str = "funasr") -> logging.Logger:
        """
        获取日志实例，如果未初始化则先初始化
        
        参数:
            name: 日志名称，默认为"funasr"
        
        返回:
            logging.Logger: 日志记录器实例
        """
        if not cls._initialized:
            cls._initialize_logger(name)
        return cls._logger
    
    @classmethod
    def _initialize_logger(cls, name: str) -> None:
        """
        初始化日志配置
        
        参数:
            name: 日志名称
        """
        # 创建logger
        cls._logger = logging.getLogger(name)
        
        # 如果logger已经配置过处理器，直接返回
        if cls._logger.handlers:
            cls._initialized = True
            return
        
        log_level = config_manager.get_app_config().get('log_level').upper()
        cls._logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # 创建文件处理器 - 将日志保存到logs目录下
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 获取当前日期，格式为YYYY-MM-DD
        today = datetime.date.today().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"{today}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        
        # 添加处理器到logger
        cls._logger.addHandler(console_handler)
        cls._logger.addHandler(file_handler)
        cls._initialized = True

# 提供便捷使用方式
logger = Logger.get_logger()

def get_logger(name: str = "funasr") -> logging.Logger:
    """
    获取指定名称的日志记录器
    
    参数:
        name: 日志名称
    
    返回:
        logging.Logger: 日志记录器实例
    """
    return Logger.get_logger(name)

# 提供便捷的日志记录函数
def info(message: str, *args, **kwargs) -> None:
    """记录INFO级别的日志"""
    logger.info(message, *args, **kwargs)

def debug(message: str, *args, **kwargs) -> None:
    """记录DEBUG级别的日志"""
    logger.debug(message, *args, **kwargs)

def warning(message: str, *args, **kwargs) -> None:
    """记录WARNING级别的日志"""
    logger.warning(message, *args, **kwargs)

def error(message: str, *args, **kwargs) -> None:
    """记录ERROR级别的日志"""
    logger.error(message, *args, **kwargs)

def critical(message: str, *args, **kwargs) -> None:
    """记录CRITICAL级别的日志"""
    logger.critical(message, *args, **kwargs)

# 记录请求信息的便捷函数
def log_request(method: str, url: str, client_ip: str) -> None:
    """
    记录请求信息
    
    参数:
        method: HTTP方法
        url: 请求URL
        client_ip: 客户端IP地址
    """
    info(f"请求: {method} {url} 客户端IP: {client_ip}")

# 记录响应信息的便捷函数
def log_response(status_code: int, process_time: float) -> None:
    """
    记录响应信息
    
    参数:
        status_code: HTTP状态码
        process_time: 处理时间(秒)
    """
    info(f"响应: {status_code} 处理时间: {process_time:.4f}秒")