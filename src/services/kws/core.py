# src\services\kws\core.py

from ...utils import debug, info, error
from .streaming import StreamingKWSService
from ...common import global_config
from funasr import AutoModel
from ..base_model_service import BaseModelService
from typing import Optional, Dict, Any
import numpy as np


class KWSService(BaseModelService):
    """
    关键词 spotting(KWS)服务类，基于FSMN模型实现
    使用单例模式确保模型只被加载一次
    提供模型管理和流式会话创建功能
    """

    def __init__(self):
        """初始化KWS服务，确保模型只被加载一次"""
        # 从配置读取
        self.kws_config = global_config.get_kws_config()
        self.reset()
        super().__init__(service_name="KWS服务", model_options=self.kws_config)

    def _load_model(self, **kwargs):
        """加载KWS模型"""
        return AutoModel(**kwargs)

    def reset(self):
        """重置状态"""
        self.cache = {}
        self.is_active = True  # 可用于状态机控制

    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        流式处理音频块，返回检测结果（如有）
        audio_chunk: float32, shape=(N,), 16kHz, 单声道, [-1, 1]
        """
        if not self.is_active:
            return None

        result = self._model.generate(
            input=audio_chunk,
            cache=self.cache,
        )
        debug(f"KWS 原始结果: {result}")
        debug(f"KWS 当前缓存块⌚️: {id(self.cache)}")

        # 解析结果
        keyword = self.parse_kws_result(result, threshold=0.1)

        return keyword 

    def parse_kws_result(self, result, threshold: float = 0.1) -> str | None:
        """
        解析 FunASR KWS 模型输出，返回唤醒词或 None
        兼容流式/非流式返回格式
        """
        # 1. 类型安全检查：确保 result 是 list 且非空
        if not isinstance(result, list) or len(result) == 0:
            error(f"KWS 结果格式错误：期望非空 list，实际为 {type(result)} = {result}")
            return None

        # 2. 获取第一个结果项
        first_item = result[0]

        # 3. 检查是否为 dict
        if not isinstance(first_item, dict):
            error(f"KWS 结果项非 dict：{type(first_item)} = {first_item}")
            return None

        # 4. 提取 text 字段
        raw_text = first_item.get("text", "").strip()
        if not raw_text:
            debug("KWS 无有效文本输出")
            return None

        # 5. 必须以 'detected ' 开头才认为是有效检测
        if not raw_text.startswith("detected "):
            debug(f"KWS 输出非检测结果: '{raw_text}'")
            return None

        # 6. 拆分并验证字段数量
        parts = raw_text.split()
        if len(parts) < 3:
            error(f"KWS 输出字段不足: {raw_text}")
            return None

        keyword = parts[1]
        try:
            score = float(parts[2])
        except ValueError:
            error(f"KWS 置信度无法转为 float: '{parts[2]}' in '{raw_text}'")
            return None

        # 7. 日志分级：高置信度 info，低置信度 debug
        log_msg = f"KWS 解析结果 → 关键词: '{keyword}', 置信度: {score:.4f}"
        if score >= threshold:
            info(log_msg + " ✅ (达到阈值)")
            return keyword
        else:
            debug(log_msg + f" ❌ (低于阈值 {threshold})")
            return None


