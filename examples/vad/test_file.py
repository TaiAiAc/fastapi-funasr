import numpy as np
from pydub import AudioSegment

# 读取你的 m4a（你确认没问题的文件）
audio = AudioSegment.from_file("processed_1760411072657_d39d7d6f.wav")
audio = audio.set_frame_rate(16000).set_channels(1)
samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
samples = samples / 32768.0  # 转为 [-1, 1]

# 全局统计
print(f"全局 min/max: {samples.min():.4f} / {samples.max():.4f}")
print(f"全局 RMS: {np.sqrt(np.mean(samples**2)):.6f}")

# 重点检查 3.5s~7.5s
start = int(3.5 * 16000)
end = int(7.5 * 16000)
seg = samples[start:end]
print(f"\n3.5s~7.5s 段:")
print(f"  min/max: {seg.min():.4f} / {seg.max():.4f}")
print(f"  RMS: {np.sqrt(np.mean(seg**2)):.6f}")
print(f"  非零比例: {np.mean(np.abs(seg) > 0.001):.2%}")