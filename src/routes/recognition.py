from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional
from ..utils import logger, info, error  # 使用统一的日志工具
from ..services.vad import vad_service  # 导入VAD服务
import soundfile as sf
import numpy as np
import io
import time
import os
import uuid  # 用于生成唯一文件名
from pydub import AudioSegment  # 导入pydub库
from ..utils import AudioConverter

# 创建语音识别相关的路由器
recognition_router = APIRouter(
    prefix="/recognition",  # 路由前缀
    tags=["语音识别"],  # API文档分类标签
    responses={404: {"description": "Not found"}},
)

@recognition_router.post("/analyze_voice")
async def analyze_voice(
    audio_file: UploadFile = File(...),
    sample_rate: Optional[int] = Form(16000),
    max_end_silence_time: Optional[int] = Form(300),
    speech_noise_thres: Optional[float] = Form(0.1),  # 降低阈值，提高敏感度
    save_processed_audio: Optional[bool] = Form(False)
):
    """
    上传音频文件并分析人声
    
    Args:
        audio_file: 要分析的音频文件
        sample_rate: 音频采样率（Hz），默认16000
        max_end_silence_time: 尾部静音超时时间（ms），默认300ms
        speech_noise_thres: 语音/噪音阈值（范围0~1），默认0.1（提高敏感度）
        save_processed_audio: 是否保存处理后的音频文件，默认True
        
    Returns:
        JSONResponse: 包含人声分析结果的响应
    """
    info(f"音频分析接口被调用，文件名: {audio_file.filename}")
    
    try:
        # 检查VAD服务是否已初始化
        if not vad_service.is_initialized():
            error("VAD模型未初始化")
            return JSONResponse(
                status_code=503,
                content={"success": False, "error": "VAD模型未初始化完成"}
            )
        
        # 读取音频文件内容
        audio_data = await audio_file.read()
        info(f"读取音频文件完成，大小: {len(audio_data)}字节")
        # 判断文件格式
        file_ext = audio_file.filename.lower().split('.')[-1]
        
        # 创建音频流并解析
        try:
            # 优先使用soundfile直接读取
            try:
                audio_stream = io.BytesIO(audio_data)
                data, sr = sf.read(audio_stream, dtype="float32")
                info(f"使用soundfile解析音频成功，采样率: {sr}Hz, 声道数: {data.ndim}")
            except Exception as sf_error:
                # 如果是m4a格式或其他soundfile不支持的格式，使用pydub处理
                if file_ext == 'm4a' or "Format not recognised" in str(sf_error):
                    info(f"尝试使用pydub处理{file_ext}格式音频")
                    # 使用pydub读取音频
                    audio_segment = AudioSegment.from_file(
                        io.BytesIO(audio_data), 
                        format=file_ext
                    )
                    
                    # 转换为16kHz单声道float32格式
                    audio_segment = audio_segment.set_frame_rate(sample_rate).set_channels(1)
                    
                    # 转换为numpy数组
                    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                    # 关键修复：正确归一化音频到[-1, 1]范围
                    data = samples / 32767.0  # ✅ 修复：使用32767作为除数
                    sr = sample_rate
                    info(f"使用pydub处理成功，采样率: {sr}Hz, 音频长度: {len(data)}样本")
                    valid_length = len(data) * 1000 / sr  # 转换为毫秒
                    info(f"音频数据范围: [{data.min():.6f}, {data.max():.6f}]")  # 添加范围日志
                else:
                    raise sf_error
            
            # 如果是多声道，转换为单声道
            if data.ndim > 1:
                data = data.mean(axis=1)
                info("已将多声道音频转换为单声道")
                
            # 保存处理后的音频文件
            processed_audio_path = None
            if save_processed_audio:
                # 确保保存目录存在
                save_dir = "processed_audio"
                os.makedirs(save_dir, exist_ok=True)
                
                # 生成唯一文件名
                timestamp = int(time.time() * 1000)
                unique_id = str(uuid.uuid4())[:8]
                base_filename = f"processed_{timestamp}_{unique_id}.wav"
                processed_audio_path = os.path.join(save_dir, base_filename)
                
                # 保存为16kHz单声道WAV文件
                sf.write(processed_audio_path, data, sample_rate, subtype='PCM_16')
                info(f"处理后的音频已保存到: {processed_audio_path}")
                
        except Exception as e:
            error(f"音频解析失败: {e}")
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"音频文件格式错误: {str(e)}"}
            )
        
            # === 音频加载完成，统一转换为 int16 用于 VAD ===
        
        try:
            vad_data = AudioConverter.to_int16(data)  # ✅ 一行搞定！
        except Exception as e:
            error(f"音频格式转换失败: {e}")
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"音频格式不支持: {str(e)}"}
            )

        # 创建VAD流进行分析
        vad_stream = vad_service.create_stream(
            max_end_silence_time=max_end_silence_time,
            speech_noise_thres=speech_noise_thres,
        )
        
        # 准备分析参数
        analyze_params = {
            "max_end_silence_time": max_end_silence_time,
            "speech_noise_thres": speech_noise_thres
        }
        info(f"开始音频分析，参数: {analyze_params}")
        
        # 测量分析时间
        start_time = time.time()
        
        all_segments = []
        
        # 将音频分割成小块进行处理
        chunk_samples = int(vad_stream.chunk_duration_ms * sample_rate / 1000)  # 600ms → 9600

        for i in range(0, len(vad_data), chunk_samples):
            chunk = vad_data[i:i + chunk_samples]
            # ✅ Padding 到 160 的整数倍（FunASR 要求）
            remainder = len(chunk) % 160
            if remainder != 0:
                chunk = np.pad(chunk, (0, 160 - remainder), mode='constant')
            vad_stream.process(chunk)
        
        # 关键：调用 finish() 处理残留语音段
        all_segments = vad_stream.finish()
        
        process_time = time.time() - start_time
        info(f"音频分析完成，耗时: {process_time:.2f}秒，检测到 {len(all_segments)} 个语音段")
        
        # 计算语音总时长
        total_voice_duration = sum(end - start for start, end in all_segments) if all_segments else 0
        audio_duration = len(data) / sample_rate * 1000  # 转换为毫秒
        
        # 返回分析结果（包含保存的音频路径）
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "filename": audio_file.filename,
                "audio_duration": round(audio_duration, 2),  # 音频总时长(ms)
                "voice_segments": all_segments,  # 语音段[[start_ms, end_ms], ...]
                "total_voice_segments": len(all_segments),  # 语音段数量
                "total_voice_duration": round(total_voice_duration, 2),  # 语音总时长(ms)
                "voice_ratio": round(total_voice_duration / audio_duration * 100, 2) if audio_duration > 0 else 0,  # 语音占比(%)
                "process_time": round(process_time, 2),  # 处理时间(秒)
                "sample_rate": sample_rate,
                "params": analyze_params,
                "processed_audio_path": processed_audio_path  # 新增字段：处理后的音频文件路径
            }
        )
        
    except Exception as e:
        error(f"音频分析过程中发生异常: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"分析过程中发生错误: {str(e)}"}
        )