# test_vad_direct.py
import soundfile as sf
import numpy as np
from funasr import AutoModel

# 1. 读取音频文件
audio, sr = sf.read("vad_example.wav", dtype="float32")  # 正确读取为 float32
if audio.ndim > 1:
    audio = audio.mean(axis=1).astype(np.float32)  # 转单声道

# 确保采样率为 16000 Hz
if sr != 16000:
    import resampy
    print(f"重采样: {sr}Hz -> 16000Hz")
    audio = resampy.resample(audio, sr, 16000)
    sr = 16000

print(f"音频长度: {len(audio)} samples, 采样率: {sr} Hz")

# 2. 使用正确的方法调用 FunASR VAD 模型
model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")

# 方法 1: 使用流式推理接口 (推荐)
print("\n=== 使用流式推理接口 ===")
cache = {}
chunk_size = 200  # ms
chunk_samples = int(chunk_size * sr / 1000)
total_chunk_num = int(len(audio) / chunk_samples) + 1

segments = []
for i in range(total_chunk_num):
    start = i * chunk_samples
    end = min((i + 1) * chunk_samples, len(audio))
    chunk = audio[start:end]
    is_final = (i == total_chunk_num - 1)
    
    # 使用 generate 方法并提供 cache
    result = model.generate(input=chunk, cache=cache, is_final=is_final, chunk_size=chunk_size)
    
    # 获取结果
    if result and len(result[0].get("value", [])) > 0:
        current_segments = result[0]["value"]
        segments.extend(current_segments)
        print(f"块 {i+1}/{total_chunk_num} 检测到语音段: {current_segments}")

print("\n合并后的语音段:", segments)

# 方法 2: 使用简化的离线接口 (如果支持)
print("\n=== 使用简化的离线接口 ===")
try:
    # 某些版本可能支持这种简化调用
    result = model.generate(input=audio)
    print("离线结果:", result)
except Exception as e:
    print(f"离线接口调用失败: {e}")