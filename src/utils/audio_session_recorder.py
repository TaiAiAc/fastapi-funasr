# src\utils\audio_session_recorder.py

import numpy as np
import os
import wave
import time
from pathlib import Path
from typing import Optional

from .logger import logger


class AudioSessionRecorder:
    """
    用于在流式语音会话中累积音频并定期保存为 WAV 文件（如每10秒）。
    每个实例代表一个独立会话（如一个 WebSocket 连接）。
    """

    def __init__(
        self,
        session_id: str,
        sample_rate: int = 16000,
        save_dir: Optional[str] = "logs",
        segment_duration_sec: int = 10,
    ):
        """
        Args:
            session_id: 会话唯一标识（如 WebSocket 连接ID）
            sample_rate: 采样率
            save_dir: 保存根目录，默认为 ./audio_sessions
            segment_duration_sec: 每段保存的时长（秒）
        """
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.segment_duration_sec = segment_duration_sec

        # 获取当天日期
        current_date = time.strftime("%Y-%m-%d", time.localtime())
        self.save_dir = Path(save_dir) / "audio_sessions" / current_date / session_id
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.buffer = np.array([], dtype=np.float32)  # 始终用 float32 内部存储
        self.segment_index = 0
        self.start_time = time.time()

        logger.info(f"🎧 开始录音会话 [{session_id}]，保存至: {self.save_dir}")

    def add_chunk(self, audio_chunk: np.ndarray):
        """添加一个音频 chunk 到缓冲区，并检查是否需要切分保存"""

        if audio_chunk.size == 0:
            return

        # 统一转为 float32 内部存储（便于处理）
        if audio_chunk.dtype == np.int16:
            audio_f32 = audio_chunk.astype(np.float32) / 32768.0
        elif audio_chunk.dtype == np.float32:
            audio_f32 = audio_chunk.copy()
        else:
            # 兜底：先转 float32
            audio_f32 = audio_chunk.astype(np.float32)
            if np.max(np.abs(audio_f32)) > 1.0:
                audio_f32 /= np.max(np.abs(audio_f32))

        self.buffer = np.concatenate([self.buffer, audio_f32])

        # 检查是否达到切分阈值
        samples_per_segment = self.sample_rate * self.segment_duration_sec
        while len(self.buffer) >= samples_per_segment:
            segment = self.buffer[:samples_per_segment]
            self._save_segment(segment)
            self.buffer = self.buffer[samples_per_segment:]

    def finalize(self):
        """结束会话，保存剩余音频"""

        if self.buffer.size > 0:
            self._save_segment(self.buffer, is_final=True)
            self.buffer = np.array([], dtype=np.float32)

        logger.info(f"⏹️ 会话 [{self.session_id}] 录音结束，共 {self.segment_index} 段")

    def _save_segment(self, audio_f32: np.ndarray, is_final: bool = False):
        """保存一段音频为 WAV（转为 int16）"""
        if audio_f32.size == 0:
            return

        # 转为 int16
        clipped = np.clip(audio_f32, -1.0, 1.0)
        wav_data = (clipped * 32767).astype(np.int16)

        filename = f"segment_{self.segment_index:03d}.wav"
        if is_final:
            filename = f"segment_{self.segment_index:03d}_final.wav"

        filepath = self.save_dir / filename

        try:
            with wave.open(str(filepath), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(wav_data.tobytes())
            logger.info(
                f"💾 保存音频段: {filepath.name} ({len(audio_f32)/self.sample_rate:.1f}s)"
            )
            self.segment_index += 1
        except Exception as e:
            logger.error(f"❌ 保存音频段失败: {e}")
