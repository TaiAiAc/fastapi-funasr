#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(
    model="iic/speech_sanm_kws_phone-xiaoyun-commands-online",
    keywords="小云小云",
    output_dir="./outputs/debug",
    device="cpu",
    chunk_size=[4, 8, 4],
    encoder_chunk_look_back=0,
    decoder_chunk_look_back=0,
)

import os

current_dir = os.path.join(os.path.dirname(__file__))
wav_path = os.path.join(current_dir, "..", "vad", "vad_example.wav")
wav_path = os.path.normpath(wav_path)
print(wav_path)


cache = {}
res = model.generate(input=wav_path, cache=cache)
print(res)
