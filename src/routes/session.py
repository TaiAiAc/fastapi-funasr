from typing import List

import numpy as np

from ..common import VADState, KWSState, ASRState
from ..services import VADStream


class VoiceSession:
    def __init__(self):
        self.vad_stream = VADStream()
        self.vad_state = VADState.IDLE

        self.kws_state = KWSState.WAITING
        self.asr_state = ASRState.IDLE

        self.audio_buffer: List[np.ndarray] = []
