# src\services\vad\stream.py

from typing import List
import numpy as np
from ...utils import debug

class StreamingVADService:
    """
    FunASR VAD 流式处理器，支持实时语音活动检测与离线分析。
    正确处理流式返回的 [-1, end] / [start, -1] 段。
    """

    def __init__(
        self,
        model,
        chunk_duration_ms: int = 600,
        merge_gap_ms: int = 50,
        max_end_silence_time: int = 800,
        speech_noise_thres: float = 0.8,
    ):
        self.model = model
        self.chunk_duration_ms = chunk_duration_ms
        self.merge_gap_ms = merge_gap_ms
        self.max_end_silence_time = max_end_silence_time
        self.speech_noise_thres = speech_noise_thres
        self.reset()

    def reset(self):
        """重置流状态"""
        self.cache = {}
        self.total_samples = 0
        self.chunk_size = self.chunk_duration_ms // 10  # 600ms → 60 chunks (10ms each)
        self._accumulated_raw_segments = []  # 原始段，可能含 -1
        self._last_vad_state = 0

    def _merge_segments(self, segments: List[List[int]]) -> List[List[int]]:
        """合并重叠或相邻的语音段"""
        if not segments:
            return []
        sorted_segs = sorted(segments, key=lambda x: x[0])
        merged = [sorted_segs[0][:]]
        for curr in sorted_segs[1:]:
            last = merged[-1]
            if curr[0] <= last[1] + self.merge_gap_ms:
                last[1] = max(last[1], curr[1])
            else:
                merged.append(curr[:])
        return merged

    def process(self, audio_chunk: np.ndarray) -> List[List[int]]:
        """
        处理一个音频块（必须为 int16）
        """
        if len(audio_chunk) == 0:
            return []

        if audio_chunk.dtype != np.int16:
            raise ValueError(f"VAD 流式输入必须是 int16，当前类型: {audio_chunk.dtype}")

        result = self.model.generate(
            input=audio_chunk,
            cache=self.cache,
            chunk_size=self.chunk_size,
            is_final=False,
            max_end_silence_time=self.max_end_silence_time,
            speech_noise_thres=self.speech_noise_thres,
        )

        segments_ms = result[0].get("value", [])
        self.total_samples += len(audio_chunk)
        self._last_vad_state = 1 if segments_ms else 0

        if segments_ms:
            self._accumulated_raw_segments.extend(segments_ms)

        debug(f"VAD 输出原始段: {segments_ms}")

        return segments_ms

    def finish(self) -> List[List[int]]:
        if self.total_samples == 0:
            return []

        # 触发 final，不传新音频
        result = self.model.generate(
            input=np.array([], dtype=np.int16),
            cache=self.cache,
            chunk_size=self.chunk_size,
            is_final=True,
            max_end_silence_time=self.max_end_silence_time,
            speech_noise_thres=self.speech_noise_thres,
        )

        final_segments = result[0].get("value", [])
        all_raw = self._accumulated_raw_segments + final_segments
        total_duration_ms = int(self.total_samples / 16.0)

        # === 关键：正确解析带 -1 的段 ===
        fixed_segments = []
        i = 0
        while i < len(all_raw):
            seg = all_raw[i]
            if len(seg) != 2:
                i += 1
                continue

            start, end = seg

            # 情况1: 完整段 [a, b]
            if start != -1 and end != -1:
                if start < end:
                    fixed_segments.append([start, end])
                i += 1

            # 情况2: [a, -1] 后跟 [-1, b] → 合并为 [a, b]
            elif start != -1 and end == -1:
                if i + 1 < len(all_raw):
                    next_seg = all_raw[i + 1]
                    if len(next_seg) == 2 and next_seg[0] == -1 and next_seg[1] != -1:
                        merged_end = next_seg[1]
                        if start < merged_end:
                            fixed_segments.append([start, merged_end])
                        i += 2  # 跳过两个段
                        continue
                # 无法配对，用当前总时长作为结束（保守）
                if start < total_duration_ms:
                    fixed_segments.append([start, total_duration_ms])
                i += 1

            # 情况3: [-1, b] 但前面没有配对 → 用 0 作为起点（罕见）
            elif start == -1 and end != -1:
                if 0 < end:
                    fixed_segments.append([0, end])
                i += 1

            # 情况4: [-1, -1] 忽略
            else:
                i += 1

        # 合并相邻段
        merged = self._merge_segments(fixed_segments)
        return merged

    def is_speech_active(self) -> bool:
        return self._last_vad_state == 1

    def get_total_duration_ms(self) -> int:
        return int(self.total_samples / 16.0)