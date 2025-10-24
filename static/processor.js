// processor.js
class AudioRecorderProcessor extends AudioWorkletProcessor {
  // 在 AudioRecorderProcessor 中累积数据
  buffer = [];
  bufferSize = 0;
  constructor() {
    super();
    this.port.onmessage = (event) => {
      // 可接收来自主线程的消息（如启动/停止）
    };
  }

  process(inputs, outputs, parameters) {
    TARGET_LENGTH = 3200; // ~200ms @16kHz\
    const input = inputs[0];
    if (input.length > 0) {
      const data = input[0].slice(); // copy
      this.buffer.push(data);
      this.bufferSize += data.length;

      if (this.bufferSize >= TARGET_LENGTH) {
        const concatenated = new Float32Array(this.bufferSize);
        let offset = 0;
        for (const buf of this.buffer) {
          concatenated.set(buf, offset);
          offset += buf.length;
        }
        this.port.postMessage(concatenated);
        this.buffer = [];
        this.bufferSize = 0;
      }
    }
    return true;
  }
}

// 注册处理器
registerProcessor("audio-recorder-processor", AudioRecorderProcessor);
