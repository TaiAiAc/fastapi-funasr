from ast import Tuple
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"


model = AutoModel(
    model=model_dir,
    trust_remote_code=False,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu",
    disable_pbar=True,
)

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
wav_path = os.path.join(current_dir, "test1.mp3")
wav_path = os.path.normpath(wav_path)  # 标准化路径

# en
res = model.generate(
    input=wav_path,
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
    ban_emo_unk=True,
)
print(res)
text = rich_transcription_postprocess(res[0]["text"])
print(text)
