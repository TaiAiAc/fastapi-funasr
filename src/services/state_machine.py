# src/services/state_machine.py

import time
from typing import List, Optional
import numpy as np
from ..common import VADState
from .event_handler import EventHandler
from ..utils import debug


class StateMachine:
    def __init__(self, handler: EventHandler):
        self.state = VADState.IDLE
        self.handler = handler
        self._current_speech_start_ms: Optional[int] = None
        self.last_active_time_ms: Optional[int] = None
        self._last_vad_end_time_ms: Optional[int] = (
            None  # 👈 新增：记录上次 VAD end 时间
        )

        # ⏱️ 超时设置
        self.silence_timeout_ms = 1000  # 长时间无语音自动结束

        # 🛡️ 防抖参数（关键！）
        self.min_speech_duration_ms = 200  # 最小有效语音长度
        self.end_debounce_ms = 600  # 语音结束后延迟确认时间
        self.start_debounce_ms = 200  # 语音开始后需持续多久才算有效
        self.continuation_window_ms = 800  # 👈 新增：VAD 结束后多长时间内新语音算延续

        # 🕒 防抖状态
        self._pending_start_time_ms: Optional[int] = None
        self._pending_end_time_ms: Optional[int] = None

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def add_audio_chunk(self, chunk: np.ndarray):
        if chunk.dtype not in (np.float32, np.int16):
            raise ValueError("音频块必须是 float32 或 int16")
        if self.state == VADState.SPEAKING:
            self.last_active_time_ms = self._now_ms()

    async def check_silence_timeout(self):
        now = self._now_ms()
        # 1. 检查是否需要确认语音结束（防抖）
        if self._pending_end_time_ms is not None:
            if now - self._pending_end_time_ms >= self.end_debounce_ms:
                self._pending_end_time_ms = None
                await self._really_handle_voice_end()

        # 2. 检查是否需要确认语音开始（防抖）
        if self._pending_start_time_ms is not None:
            if now - self._pending_start_time_ms >= self.start_debounce_ms:
                # 确认是有效语音，正式进入 SPEAKING
                self._pending_start_time_ms = None
                self.state = VADState.SPEAKING
                self._current_speech_start_ms = now  # 👈 修正：这里应为 now
                self.last_active_time_ms = now
                await self.handler.on_voice_start()

        # 3. 原有超时逻辑（长时间无活动）
        if self.state == VADState.SPEAKING and self.last_active_time_ms is not None:
            if now - self.last_active_time_ms > self.silence_timeout_ms:
                await self._schedule_voice_end()

    async def update_vad_result(self, vad_segments: List[List[int]]):
        now = self._now_ms()
        has_start = False
        has_end = False

        for seg in vad_segments:
            if len(seg) != 2:
                continue
            start, end = seg

            if start != -1 and end == -1:
                if start < 100:
                    continue
                has_start = True
            elif end != -1:
                has_end = True

        # 🔴 先处理 end（更新 _last_vad_end_time_ms）
        if has_end:
            self._last_vad_end_time_ms = now
            # 注意：不立即结束，等防抖或超时

        # 🟢 再处理 start
        if has_start:
            # 如果已在说话，忽略
            if self.state == VADState.SPEAKING:
                return

            # 如果已有 pending start，跳过
            if self._pending_start_time_ms is not None:
                return

            # 👇 新增：检查是否在“语音延续窗口”内
            if (
                self._last_vad_end_time_ms is not None
                and now - self._last_vad_end_time_ms < self.continuation_window_ms
            ):
                # 直接进入 SPEAKING，不走 start 防抖（因为是延续）
                self.state = VADState.SPEAKING
                # 保持原有的 _current_speech_start_ms（不重置！）
                # 如果之前已结束，这里需要恢复 start 时间？
                # 但通常 _current_speech_start_ms 还在，因为还没真正结束
                self.last_active_time_ms = now
                debug("🔁 语音延续（跳过 start 防抖）")
                return

            # 否则，正常走 start 防抖流程
            self._pending_start_time_ms = now
            debug(f"⏳ 计划开始语音（防抖中），时间: {now}")

        # ⚠️ 注意：如果只有 end 没有 start，也要计划结束
        if has_end and not has_start:
            if self.state == VADState.SPEAKING:
                await self._schedule_voice_end()
            elif self._pending_start_time_ms is not None:
                # 处理“start 后立即 end”的情况
                duration = now - self._pending_start_time_ms
                if duration < self.min_speech_duration_ms:
                    self._pending_start_time_ms = None
                    debug(f"🔇 语音太短 ({duration}ms)，已忽略")
                else:
                    # 先确认 start，再结束
                    self._pending_start_time_ms = None
                    self.state = VADState.SPEAKING
                    self._current_speech_start_ms = now - duration
                    self.last_active_time_ms = now
                    await self.handler.on_voice_start()
                    await self._schedule_voice_end()

    async def _schedule_voice_end(self):
        """计划结束语音（带防抖）"""
        if self.state != VADState.SPEAKING:
            return
        self._pending_end_time_ms = self._now_ms()
        debug(f"⏳ 计划结束语音（防抖中），时间: {self._pending_end_time_ms}")

    async def _really_handle_voice_end(self):
        """真正结束语音"""
        if self.state != VADState.SPEAKING:
            return

        if self._current_speech_start_ms is not None:
            duration = self._now_ms() - self._current_speech_start_ms
            if duration < self.min_speech_duration_ms:
                debug(f"🔇 语音太短 ({duration}ms)，结束时忽略")
            else:
                await self.handler.on_voice_end()
                debug("✅ 语音已真正结束")

        # 重置状态（无论是否太短）
        self.state = VADState.IDLE
        self._current_speech_start_ms = None
        self.last_active_time_ms = None
        self._last_vad_end_time_ms = None  # 👈 重置

    async def interrupt(self):
        self._pending_start_time_ms = None
        self._pending_end_time_ms = None
        self._last_vad_end_time_ms = None  # 👈 重置
        if self.state == VADState.SPEAKING:
            self.state = VADState.IDLE
            self._current_speech_start_ms = None
            self.last_active_time_ms = None
            await self.handler.on_vad_interrupt()

    def reset(self):
        self.state = VADState.IDLE
        self._current_speech_start_ms = None
        self.last_active_time_ms = None
        self._pending_start_time_ms = None
        self._pending_end_time_ms = None
        self._last_vad_end_time_ms = None  # 👈 重置
