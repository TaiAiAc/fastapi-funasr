目前项目场景是数字人实时对话场景
1. 用pdm管理依赖
2. 使用fastapi框架
3. 用funasr1.2.7模型
4. 模型配置
    1. vad模型是iic/speech_fsmn_vad_zh-cn-16k-common-pytorch
    2. kws是iic/speech_charctc_kws_phone-xiaoyun
    3. asr模型是iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online
5. 都需要支持实时流式调用
6. 都需要支持打断语音唤醒
7. vad检测出人声后送给kws进行关键词识别
8. 触发唤醒以后 开始进行asr识别
9. asr识别过程中如果检测到唤醒关键词 也需要打断asr识别 进行新的状态流转 
10. 代码目录结构
    - static\index.html
    - src\routes\funasr.py
    - src\common\states.py
    - src\services\state_machine.py
    - src\services\session_handler.py
    - src\services\base_model_service.py
    - src\services\vad\core.py
    - src\services\vad\streaming.py
    - src\services\kws\core.py
    - src\services\kws\streaming.py
    - src\services\asr\core.py
    - src\services\asr\streaming.py
谨记这就是你的上下文 给出的回复要符合我的代码实现
