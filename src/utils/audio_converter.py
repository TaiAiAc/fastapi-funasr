import numpy as np
from typing import Union
from .logger import info, warning  # 假设你有统一日志模块


class AudioConverter:
    """
    音频数据格式转换工具类，用于在 float32 [-1,1] 和 int16 [-32768, 32767] 之间安全转换。
    特别适用于 FunASR 流式 VAD（要求 int16）与通用音频处理（常用 float32）之间的桥接。
    """

    @staticmethod
    def to_int16(audio: np.ndarray, source_dtype: str = "auto") -> np.ndarray:
        """
        将音频数组安全转换为 int16 格式，适用于 FunASR 流式 VAD 输入。

        Args:
            audio: 输入音频数组
            source_dtype: 源数据类型，可选 "float", "int", "auto"（默认自动检测）

        Returns:
            np.ndarray: int16 格式的音频数组

        Raises:
            ValueError: 当无法安全转换时
        """
        if audio.size == 0:
            return audio.astype(np.int16)

        if source_dtype == "auto":
            if np.issubdtype(audio.dtype, np.floating):
                source_dtype = "float"
            elif np.issubdtype(audio.dtype, np.integer):
                source_dtype = "int"
            else:
                raise ValueError(f"不支持的音频数据类型: {audio.dtype}")

        if source_dtype == "float":
            # float32/64 [-1.0, 1.0] → int16 [-32768, 32767]
            if audio.min() < -1.0 or audio.max() > 1.0:
                warning("检测到 float 音频超出 [-1, 1] 范围，将进行裁剪")
                audio = np.clip(audio, -1.0, 1.0)
            int16_audio = (audio * 32767.0).astype(np.int16)
         
            return int16_audio

        elif source_dtype == "int":
            # int32/int16 → int16（直接转换，可能截断）
            if audio.dtype == np.int16:
                info("音频已是 int16 格式，无需转换")
                return audio.copy()
            else:
                # 检查是否超出 int16 范围
                if audio.min() < -32768 or audio.max() > 32767:
                    warning("整型音频超出 int16 范围，将进行裁剪")
                    audio = np.clip(audio, -32768, 32767)
                int16_audio = audio.astype(np.int16)
                info(f"音频已从 {audio.dtype} 转换为 int16")
                return int16_audio

        else:
            raise ValueError(f"不支持的 source_dtype: {source_dtype}")

    @staticmethod
    def to_float32(audio: np.ndarray) -> np.ndarray:
        """将 int16 音频转为 float32 [-1, 1]，用于保存或通用处理"""
        if audio.dtype == np.float32:
            return audio.copy()
        elif np.issubdtype(audio.dtype, np.integer):
            float_audio = audio.astype(np.float32) / 32767.0
            return np.clip(float_audio, -1.0, 1.0)
        else:
            raise ValueError(f"无法将 {audio.dtype} 转为 float32")
