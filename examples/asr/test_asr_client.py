# test_asr_client.py
import asyncio
import websockets
import json
import numpy as np
import soundfile as sf

# 配置
CHUNK_DURATION_MS = 600
SAMPLE_RATE = 16000  # 模型采样率


async def send_audio_chunks(uri: str, wav_path: str):
    async with websockets.connect(uri) as websocket:
        print(f"✅ 已连接到 ASR 服务: {uri}")

        # 读取音频文件
        audio, sr = sf.read(wav_path, dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(f"音频采样率必须是 {SAMPLE_RATE}Hz，当前为 {sr}Hz")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)  # 转为单声道
        print(f"🔊 音频长度: {len(audio)/sr:.2f} 秒，总点数: {len(audio)}")
        print(
            f"Audio stats: min={audio.min():.6f}, max={audio.max():.6f}, std={audio.std():.6f}"
        )

        # 计算每块点数
        chunk_size = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 9600 点
        total_chunks = int(np.ceil(len(audio) / chunk_size))

        # 分块发送
        for i in range(total_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(audio))
            chunk = audio[start:end]
            print("Chunk dtype:", chunk.dtype)
            print("Is native byte order?", chunk.dtype.isnative)
            chunk = np.ascontiguousarray(chunk, dtype=np.float32)
            print("ascontiguousarray Chunk dtype:", chunk.dtype)
            print("ascontiguousarray Is native byte order?", chunk.dtype.isnative)
            await websocket.send(chunk.tobytes())

            print(f"📤 发送第 {i+1}/{total_chunks} 块，长度: {len(chunk)}")

            # 接收结果（非阻塞：尝试接收，但不等太久）
            try:
                res = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                msg = json.loads(res)
                if msg["type"] == "asr_event":
                    print(f"💬 实时结果: {msg['text']}")
                elif msg["type"] == "asr_result":  # 兼容你当前返回的 "result"
                    print(f"💬 中间结果: {msg['text']}")
            except asyncio.TimeoutError:
                pass  # 没有立即返回结果，继续发下一块

        # 发送 stop 触发最终结果
        await websocket.send(json.dumps({"type": "stop"}))
        print("⏹️ 发送 stop，等待最终结果...")

        # 接收最终结果
        final_res = await websocket.recv()
        final_msg = json.loads(final_res)
        print(f"✅ 最终识别结果: {final_msg.get('data', '')}")
        await websocket.close()
        print("✅ WebSocket 连接已关闭")


if __name__ == "__main__":
    uri = f"ws://localhost:8000/test_asr/ws"
    # 修复路径问题，使用正确的相对路径
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    wav_path = os.path.join(current_dir, "test.wav")
    wav_path = os.path.normpath(wav_path)  # 标准化路径
    asyncio.run(send_audio_chunks(uri, wav_path))
