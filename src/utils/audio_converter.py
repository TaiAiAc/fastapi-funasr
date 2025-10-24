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
    def int16_to_float32(
        x: Union[np.ndarray, list], normalize: bool = True, allow_clipping: bool = False
    ) -> np.ndarray:
        """
        将 int16 音频数据安全地转换为 float32。

        Args:
            x (np.ndarray or list): 输入的 int16 音频数据（1D 或 2D）
            normalize (bool): 是否归一化到 [-1.0, 1.0] 范围（推荐 True）
            allow_clipping (bool):
                - 若为 False（默认），当输入包含 -32768 时，除以 32768.0 会导致 -1.0（安全）
                - 若为 True，使用 32767.0 作为分母（可能导致 +1.00003 溢出，需配合 clip）

        Returns:
            np.ndarray: float32 类型的音频数组，范围通常为 [-1.0, 1.0]

        Examples:
            >>> audio_int16 = np.array([0, 16384, -32768], dtype=np.int16)
            >>> audio_float = int16_to_float32(audio_int16)
            >>> print(audio_float)
            [ 0.      0.5    -1.    ]
        """
        if isinstance(x, list):
            x = np.array(x, dtype=np.int16)
        elif not isinstance(x, np.ndarray):
            raise TypeError(f"输入必须是 list 或 np.ndarray，实际类型: {type(x)}")

        if x.dtype != np.int16:
            # 宽松处理：允许其他整数类型，但警告
            if np.issubdtype(x.dtype, np.integer):
                x = x.astype(np.int16)
            else:
                raise ValueError(f"输入数组 dtype 应为 int16，实际为 {x.dtype}")

        # 转为 float32 避免中间计算溢出
        x_float = x.astype(np.float32)

        if normalize:
            if allow_clipping:
                # 使用 32767.0，但需注意 -32768 / 32767 ≈ -1.00003
                x_float = x_float / 32767.0
                # 可选：裁剪到 [-1, 1]
                # x_float = np.clip(x_float, -1.0, 1.0)
            else:
                # 推荐方式：使用 32768.0，确保 -32768 → -1.0，32767 → ~0.99997
                x_float = x_float / 32768.0

        return x_float
