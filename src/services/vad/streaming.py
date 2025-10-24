# src\services\vad\streaming.py

from typing import List
import numpy as np
from ...utils import debug, log_audio_input, AudioConverter
from typing import Dict, Any


class StreamingVADService:
    """
    FunASR VAD 流式处理器，支持实时语音活动检测与离线分析。
    完全按照FunASR流式VAD的正确使用方式实现，优化了段处理逻辑。
    """

    def __init__(
        self,
        model,
        generate_options: Dict[str, Any],
        merge_gap_ms: int = 50,
    ):
        self.model = model
        self.generate_options = generate_options
        self.merge_gap_ms = merge_gap_ms
        self.reset()

    def reset(self):
        """重置流状态"""
        self.cache = {}
        self.total_samples = 0
        self._last_vad_state = 0
        self._accumulated_raw_segments = []
        self._has_warmup = False

    def _merge_segments(self, segments: List[List[int]]) -> List[List[int]]:
        """合并重叠或相邻的完整语音段（仅用于最终输出）

        Args:
            segments: 已闭合的语音段列表 [[start, end], ...]

        Returns:
            合并后的语音段列表
        """
        if not segments:
            return []

        # 只处理 start < end 的有效段
        valid_segs = [s for s in segments if s[0] < s[1]]
        if not valid_segs:
            return []

        # 按起始时间排序
        sorted_segs = sorted(valid_segs, key=lambda x: x[0])

        merged = [sorted_segs[0][:]]
        for curr in sorted_segs[1:]:
            last = merged[-1]
            # 如果当前段的起始时间小于等于上一段的结束时间+合并间隔，则合并
            if curr[0] <= last[1] + self.merge_gap_ms:
                last[1] = max(last[1], curr[1])
            else:
                merged.append(curr[:])

        return merged

    def _resolve_vad_segments(
        self, segments: List[List[int]], total_duration_ms: int
    ) -> List[List[int]]:
        """解析 FunASR 流式 VAD 输出，生成完整 [start, end] 段

        Args:
            segments: VAD原始输出段列表
            total_duration_ms: 总音频时长（毫秒）

        Returns:
            解析后的完整语音段列表
        """
        resolved = []
        for seg in segments:
            if not (isinstance(seg, list) and len(seg) == 2):
                continue

            start, end = seg

            if start != -1 and end != -1:
                # 完整段：确保start < end
                if start < end:
                    resolved.append([start, end])
            elif start != -1 and end == -1:
                # 未闭合段：用总时长闭合（确保 start 有效）
                if start < total_duration_ms:
                    resolved.append([start, total_duration_ms])
            elif start == -1 and end != -1:
                # 起始未知：用 0 作为起点（罕见情况）
                if 0 < end:
                    resolved.append([0, end])
            # 忽略 [-1, -1] 段

        return resolved

    def process(self, audio_int16: np.ndarray) -> List[List[int]]:
        """
        处理一个音频块（必须为 int16）
        关键点：每个chunk直接送入，动态计算chunk_size，保留原始VAD输出

        Args:
            audio_int16: 音频数据（int16格式）

        Returns:
            VAD检测到的原始段列表（包含[start, -1]等未闭合段）
        """

        # 检查当前chunk的大小
        # chunk_size_ms = int(len(audio_int16) / 16.0)
        # debug(f"当前chunk大小: {chunk_size_ms}ms")
        # log_audio_input(
        #     audio_int16, expected_format="int16", sample_rate=16000, name="VAD"
        # )

        chunk_float32 = AudioConverter.int16_to_float32(audio_int16)
 
        try:
            # 直接处理每个chunk，不累积
            result = self.model.generate(
                **self.generate_options,
                input=chunk_float32,
                cache=self.cache,  # 连续传递cache，保持模型状态
                is_final=False,
            )
        except Exception as e:
            error(f"VAD 处理失败: {e}")
            self.reset()
            return []

        debug(f"VAD 原始输出: {result}")
        segments_ms = result[0].get("value", [])

        self.total_samples += len(chunk_float32)

        # 仅做格式校验，保留所有合法段（包括 [start, -1]）
        valid_segments = [
            seg for seg in segments_ms if isinstance(seg, list) and len(seg) == 2
        ]

        # 更新语音活跃状态：存在未闭合段即为活跃
        self._last_vad_state = 1 if any(end == -1 for _, end in valid_segments) else 0

        # 保存原始段用于最终处理
        if valid_segments:
            self._accumulated_raw_segments.extend(valid_segments)

        return valid_segments

    def finish(self) -> List[List[int]]:
        """处理流结束时的最终VAD状态

        Returns:
            合并后的完整语音段列表
        """
        if self.total_samples == 0:
            return []

        # 触发final调用，传递空音频
        result = self.model.generate(
            **self.generate_options,
            input=np.array([], dtype=np.int16),
            cache=self.cache,
            is_final=True,
        )

        final_segments = result[0].get("value", [])

        # 过滤final段中的异常项
        valid_final = [
            seg for seg in final_segments if isinstance(seg, list) and len(seg) == 2
        ]

        # 合并所有累积的段和final段
        all_raw = self._accumulated_raw_segments + valid_final
        total_duration_ms = int(self.total_samples / 16.0)

        # 先解析为完整段，再合并
        resolved = self._resolve_vad_segments(all_raw, total_duration_ms)
        merged = self._merge_segments(resolved)

        return merged

    def is_speech_active(self) -> bool:
        """判断当前是否处于语音活动状态

        Returns:
            True表示处于语音活动状态，False表示静音状态
        """
        return self._last_vad_state == 1

    def get_total_duration_ms(self) -> int:
        """获取处理的总音频时长（毫秒）

        Returns:
            总音频时长（毫秒）
        """
        return int(self.total_samples / 16.0)
