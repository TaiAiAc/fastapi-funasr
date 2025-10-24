class AudioBuffer:
    def __init__(self, target_len=1600, overlap_len=320):  # 100ms @16kHz
        self.buf = np.array([], dtype=np.int16)
        self.target = target_len
        self.overlap = overlap_len

    def add(self, audio: np.ndarray):
        self.buf = np.concatenate([self.buf, audio.astype(np.int16)])

    def get_chunks(self):
        chunks = []
        while len(self.buf) >= self.target:
            chunks.append(self.buf[: self.target])
            self.buf = self.buf[self.target :]
        return chunks

    def clear(self):
        self.buf = np.array([], dtype=np.int16)

    @property
    def current_duration_ms(self):
        """当前缓冲区中的音频时长（毫秒）"""
        return len(self.buf) / 16  # 16kHz采样率，每毫秒16个采样点

    def get_remaining(self):
        """获取缓冲区中剩余的数据（不足一个完整块时）"""
        if len(self.buf) > 0:
            return self.buf.copy()
        return np.array([], dtype=np.int16)
