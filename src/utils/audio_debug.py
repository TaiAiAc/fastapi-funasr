# utils/audio_debug.py

import numpy as np
from .logger import logger
from typing import Optional, Literal


# 预定义音频格式规范
AUDIO_FORMAT_SPECS = {
    "float32": {
        "dtype": np.float32,
        "range": (-1.0, 1.0),
        "description": "32-bit float, normalized to [-1.0, 1.0]",
    },
    "int16": {
        "dtype": np.int16,
        "range": (-32768, 32767),
        "description": "16-bit signed integer",
    },
    "int8": {
        "dtype": np.int8,
        "range": (-128, 127),
        "description": "8-bit signed integer (rare)",
    },
}


def log_audio_input(
    audio: np.ndarray,
    name: str = "audio",
    sample_rate: int = 16000,
    expected_format: Optional[Literal["float32", "int16", "int8"]] = None,
) -> None:
    """
    打印音频输入的详细调试信息，并校验是否符合预期格式。
    即使未指定 expected_format，也会尝试推断并给出建议。
    """
    # === 基础类型与形状校验 ===
    if not isinstance(audio, np.ndarray):
        logger.error(f"[{name}] 输入类型错误：期望 np.ndarray，实际为 {type(audio)}")
        return

    if audio.ndim != 1:
        logger.error(f"[{name}] 音频维度错误：shape={audio.shape}，应为 1D 数组")
        return

    length_samples = len(audio)
    if length_samples == 0:
        logger.error(f"[{name}] 音频长度为 0，无法分析")
        return

    # === 安全计算 RMS（避免 int 溢出）===
    if audio.dtype.kind == 'i':  # 整数类型
        # 转为 float64 前先归一化到 [-1, 1] 范围（以 int16 为基准）
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float64) / 32768.0
        elif audio.dtype == np.int8:
            audio_float = audio.astype(np.float64) / 128.0
        else:
            # 兜底：用最大值归一化
            max_val = np.iinfo(audio.dtype).max
            audio_float = audio.astype(np.float64) / float(max_val)
    else:
        audio_float = audio.astype(np.float64)

    rms = float(np.sqrt(np.mean(np.square(audio_float))))

    # === 基础统计信息 ===
    duration_ms = length_samples * 1000.0 / sample_rate if sample_rate > 0 else float("nan")
    dtype = audio.dtype
    min_val = float(audio.min())
    max_val = float(audio.max())

    # 能量等级判断
    if rms < 1e-6:
        energy_level = "静音"
    elif rms < 0.01:
        energy_level = "极低"
    elif rms < 0.05:
        energy_level = "低"
    else:
        energy_level = "正常"

    base_msg = (
        f"[{name}] shape={audio.shape}, dtype={dtype}, "
        f"duration={duration_ms:.1f}ms, "
        f"range=[{min_val:.3f}, {max_val:.3f}], "
        f"RMS={rms:.5f}, 能量: {energy_level}"
    )

    errors = []
    warnings = []

    # === 自动推断可能的格式 ===
    inferred_format = None
    if dtype == np.float32 or dtype == np.float64:
        if -1.1 <= min_val and max_val <= 1.1:
            inferred_format = "float32"
        else:
            # 值太大，可能误传了 int16 的二进制
            inferred_format = "疑似 int16（但以 float 解析）"
    elif dtype == np.int16:
        if abs(max_val) > 1000:
            inferred_format = "int16"
        else:
            inferred_format = "疑似 float32（未缩放为整数）"
    elif dtype == np.int8:
        inferred_format = "int8"

    # === 如果指定了 expected_format，进行严格校验 ===
    if expected_format:
        spec = AUDIO_FORMAT_SPECS.get(expected_format)
        if spec is None:
            logger.error(f"[{name}] 不支持的 expected_format: '{expected_format}'")
            return

        expected_dtype = spec["dtype"]
        low, high = spec["range"]

        # 1. 数据类型检查
        if dtype != expected_dtype:
            errors.append(f"数据类型应为 {expected_dtype}，但实际为 {dtype}")

        # 2. 数值范围检查（允许微小容差）
        tol = 1e-3
        if min_val < low - tol or max_val > high + tol:
            errors.append(f"数值超出 {expected_format} 有效范围 [{low}, {high}]")

        # 3. 特定误用检测
        if expected_format == "float32":
            if abs(max_val) > 2.0:
                warnings.append("幅值 >2.0，疑似将 int16 数据误当作 float32（应除以 32768）")
        elif expected_format == "int16":
            if abs(max_val) < 100:
                warnings.append("int16 幅值 <100，疑似 float32 未乘 32768 转为整数")

    # === 通用合理性检查 ===
    SILENCE_RMS_THRESHOLD = 5e-4
    if rms < SILENCE_RMS_THRESHOLD:
        warnings.append(f"RMS={rms:.5f} 过低，可能为静音（阈值: {SILENCE_RMS_THRESHOLD})")
    elif rms < 0.01:
        warnings.append(f"语音能量偏低 (RMS={rms:.5f})，可能影响 VAD/KWS 检测效果")

    # === 输出日志 ===
    full_msg = base_msg
    if inferred_format and not expected_format:
        full_msg += f" | 推断格式: {inferred_format}"

    if errors:
        logger.error(f"{full_msg} | ❌ 错误: {'; '.join(errors)}")
    elif warnings:
        logger.warning(f"{full_msg} | ⚠️ 警告: {'; '.join(warnings)}")
    else:
        logger.debug(full_msg)