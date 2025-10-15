from funasr import AutoModel

model = AutoModel(
    model="iic/speech_charctc_kws_phone-xiaoyun",
    keywords="小云小云",
    output_dir="./outputs/debug",
    device='cpu'
)

test_wav = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testset/kws_xiaoyunxiaoyun.wav"

res = model.generate(input=test_wav, cache={},)
print(res)