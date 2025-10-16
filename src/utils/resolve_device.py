import torch
from typing import Literal

def resolve_device(device_config: str) -> Literal["cuda", "cpu"]:
    """
    根据配置字符串解析实际使用的设备（cuda / cpu）
    
    Args:
        device_config (str): 配置值，支持 "auto"、"cuda"、"cpu"
    
    Returns:
        str: 实际设备名称，"cuda" 或 "cpu"
    """
    from . import error 

    device_config = device_config.lower().strip()
    
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device_config in ("cuda", "cpu"):
        return device_config
    else:
        error(f"无效的设备配置: '{device_config}'，将回退到自动检测")
        return "cuda" if torch.cuda.is_available() else "cpu"