import numpy as np

class SimpleVoiceActivity:
    def __init__(self, energy_threshold=0.02, frame_len=320):  # 20ms @16kHz
        self.energy_threshold = energy_threshold
        self.frame_len = frame_len
        self.silence_count = 0
        self.is_active = False

    def is_voice_start(self, audio_chunk: np.ndarray) -> bool:
        # 滑动窗口检测
        for i in range(0, len(audio_chunk) - self.frame_len, self.frame_len):
            frame = audio_chunk[i:i+self.frame_len]
            energy = np.mean(frame ** 2)
            if energy > self.energy_threshold:
                if not self.is_active:
                    self.is_active = True
                    return True  # 检测到起始！
                self.silence_count = 0
                break
            else:
                self.silence_count += 1
                if self.silence_count > 10:  # 连续 200ms 静音
                    self.is_active = False
        return False