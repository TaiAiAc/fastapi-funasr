# src\services\vad\streaming.py

from typing import List
import numpy as np
from ...utils import debug

class StreamingVADService:
    """
    FunASR VAD 流式处理器，支持实时语音活动检测与离线分析。
    完全按照FunASR流式VAD的正确使用方式实现，优化了段处理逻辑。
    """

    def __init__(
        self,
        model,
        merge_gap_ms: int = 50,
        max_end_silence_time: int = 800,
        speech_noise_thres: float = 0.92,
    ):
        self.model = model
        self.merge_gap_ms = merge_gap_ms
        self.max_end_silence_time = max_end_silence_time
        self.speech_noise_thres = speech_noise_thres
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

    def _resolve_vad_segments(self, segments: List[List[int]], total_duration_ms: int) -> List[List[int]]:
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

    def process(self, audio_chunk: np.ndarray) -> List[List[int]]:
        """
        处理一个音频块（必须为 int16）
        关键点：每个chunk直接送入，动态计算chunk_size，保留原始VAD输出
        
        Args:
            audio_chunk: 音频数据（int16格式）
        
        Returns:
            VAD检测到的原始段列表（包含[start, -1]等未闭合段）
        """
        if len(audio_chunk) == 0:
            return []
        
        if audio_chunk.dtype != np.int16:
            raise ValueError(f"VAD 流式输入必须是 int16，当前类型: {audio_chunk.dtype}")
        
            # 预热：确保模型 cache 正确初始化
        if not self._has_warmup:
            warmup_chunk = np.zeros_like(audio_chunk)  # 同长度静音
            try:
                self.model.generate(
                    input=warmup_chunk,
                    cache=self.cache,
                    chunk_size=max(1, int(round(len(warmup_chunk) / 160))),  # 160 = 10ms * 16kHz
                    is_final=False,
                    max_end_silence_time=self.max_end_silence_time,
                    speech_noise_thres=self.speech_noise_thres,
                )
                self._has_warmup = True
                debug("VAD 模型预热完成")
            except Exception as e:
                debug(f"VAD 预热失败（可能影响首块检测）: {e}")
                self._has_warmup = True  # 避免重复尝试

        # 根据实际音频长度动态计算chunk_size（帧数），使用四舍五入提高准确性
        chunk_ms = len(audio_chunk) / 16.0  # 计算音频chunk的时长（毫秒）
        chunk_size = max(1, int(round(chunk_ms / 10)))  # 每帧10ms，至少1帧
        
        debug(f"处理音频块: len={len(audio_chunk)}, chunk_ms={chunk_ms:.1f}ms, chunk_size={chunk_size}")
        
        # 直接处理每个chunk，不累积
        result = self.model.generate(
            input=audio_chunk,
            cache=self.cache,  # 连续传递cache，保持模型状态
            chunk_size=chunk_size,
            is_final=False,
            max_end_silence_time=self.max_end_silence_time,
            speech_noise_thres=self.speech_noise_thres,
        )

        segments_ms = result[0].get("value", [])
        self.total_samples += len(audio_chunk)
        
        # 仅做格式校验，保留所有合法段（包括 [start, -1]）
        valid_segments = [
            seg for seg in segments_ms
            if isinstance(seg, list) and len(seg) == 2
        ]
        
        # 更新语音活跃状态：存在未闭合段即为活跃
        self._last_vad_state = 1 if any(end == -1 for _, end in valid_segments) else 0
        
        # 保存原始段用于最终处理
        if valid_segments:
            self._accumulated_raw_segments.extend(valid_segments)
        
        debug(f"VAD 输出段: {valid_segments}")
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
            input=np.array([], dtype=np.int16),
            cache=self.cache,
            chunk_size=1,  # 任意小值
            is_final=True,
            max_end_silence_time=self.max_end_silence_time,
            speech_noise_thres=self.speech_noise_thres,
        )
        
        final_segments = result[0].get("value", [])
        
        # 过滤final段中的异常项
        valid_final = [
            seg for seg in final_segments
            if isinstance(seg, list) and len(seg) == 2
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