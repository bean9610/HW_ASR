# coding=utf-8
#import hiai
#import hiai.nn_tensor_lib import DataType

import os
import numpy as np
import time

from pcm2wav import pcm2wav
from get_features import RecognizeSpeech_FromFile
from make_input_tensor import make_input_tensor

# 定义文件获取路径
speech_recog_model = 'speech_recog_model/ASR_sample.om' # 语音识别的声学模型
speech_voice_path = 'speech_voice/01.pcm' # 开发板保存的语音路径

def GetDataSet(speech_voice_path):
    # 将pcm数据转换为wav
    wave_path = pcm2wav(speech_voice_path) # 已完成

    # 读取wav音频特征
    features = RecognizeSpeech_FromFile(wave_path) # 已完成

    # 将wav音频特征转换为模型输入向量
    input_tensor = make_input_tensor(features) # 未完成！！！

    return input_tensor

ret = GetDataSet(speech_voice_path)

