import os
import datetime
import logging
import sys

# 检查是否为Windows环境并添加兼容性支持
class ColoredFormatter(logging.Formatter):
    """
    带颜色的日志格式化器
    为不同日志级别设置不同的颜色，基于Python标准库实现
    """
    # 定义不同日志级别对应的ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[92m',    
        'INFO': '\033[96m',   
        'WARNING': '\033[93m', 
        'ERROR': '\033[91m',  
        'CRITICAL': '\033[91m\033[1m', 
    }
    RESET_CODE = '\033[0m'
    
    # 对于Windows cmd.exe的特殊处理
    if sys.platform.startswith('win'):
        try:
            # 尝试启用Windows控制台的ANSI支持
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            # 如果启用失败，则禁用颜色输出
            COLORS = {}
            RESET_CODE = ''

    def format(self, record):
        # 先调用父类方法获取格式化后的消息
        formatted_message = super().format(record)
        
        # 根据日志级别添加颜色
        level_name = record.levelname
        if level_name in self.COLORS:
            return f"{self.COLORS[level_name]}{formatted_message}{self.RESET_CODE}"
        else:
            return formatted_message

class Logger:
    """
    统一日志工具类，提供简单统一的日志功能
    """
    _instance = None
    _logger = None
    _initialized = False

    @classmethod
    def get_logger(cls, name: str = "voice") -> logging.Logger:
        """
        获取日志实例，如果未初始化则先初始化
        
        参数:
            name: 日志名称，默认为"voice"
        
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
        
        # 清除已有的处理器，避免重复输出
        if cls._logger.handlers:
            cls._logger.handlers.clear()
        
        # 设置日志级别
        log_level = os.getenv("APP_LOG_LEVEL", "DEBUG")
        cls._logger.setLevel(log_level)
        
        # 设置日志格式
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # 创建控制台处理器 - 使用彩色格式化器
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter(log_format)
        console_handler.setFormatter(console_formatter)
        
        # 创建文件处理器 - 将日志保存到logs目录下
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 获取当前日期，格式为YYYY-MM-DD
        today = datetime.date.today().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"{today}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter(log_format)  # 文件日志不使用彩色格式
        file_handler.setFormatter(file_formatter)
        
        # 添加处理器到logger
        cls._logger.addHandler(console_handler)
        cls._logger.addHandler(file_handler)
        
        # 防止日志消息向上传播到父记录器（避免重复输出）
        cls._logger.propagate = False
        cls._initialized = True

# 提供便捷使用方式
def get_logger(name: str = "voice") -> logging.Logger:
    """
    获取指定名称的日志记录器
    
    参数:
        name: 日志名称
    
    返回:
        logging.Logger: 日志记录器实例
    """
    return Logger.get_logger(name)

# 初始化默认logger实例
logger = get_logger()

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

