# coding=utf-8
import numpy as np
import wave
from scipy.fftpack import fft

x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1) ) # 汉明窗
AUDIO_FEATURE_LENGTH = 200

def read_wav_data(filename):
    wav=wave.open(filename,"rb")
    num_frame=wav.getnframes()
    num_channel=wav.getnchannels()
    framerate=wav.getframerate()
    num_sample_width=wav.getsampwidth()
    str_data=wav.readframes(num_frame)
    wav.close()
    wave_data=np.fromstring(str_data,dtype=np.short)
    wave_data.shape=-1,num_channel
    wave_data=wave_data.T
    #print("ks",framerate)
    return wave_data,framerate

def GetFrequencyFeature3(wavsignal, fs):
    if (16000 != fs):
        raise ValueError(
            '[Error] ASRT currently only supports wav audio files with a sampling rate of 16000 Hz, but this audio is ' + str(
                fs) + ' Hz. ')

    # wav波形 加时间窗以及时移10ms
    time_window = 25  # 单位ms
    window_length = fs / 1000 * time_window  # 计算窗长度的公式，目前全部为400固定值
    #print window_length
    wav_arr = np.array(wavsignal)
    # wav_length = len(wavsignal[0])
    wav_length = wav_arr.shape[1]
    range0_end = int(float(len(wavsignal[0])) / fs * 1000 - time_window) // 10  # 计算循环终止的位置，也就是最终生成的窗数 978
    data_input = np.zeros((range0_end, 200), dtype=np.float)  # 用于存放最终的频率特征数据
    data_line = np.zeros((1, 400), dtype=np.float)
    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400

        data_line = wav_arr[0, p_start:p_end]

        data_line = data_line * w  # 加窗

        data_line = np.abs(fft(data_line)) / wav_length

        data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的

    # print(data_input.shape)
    data_input = np.log(data_input + 1)
    return data_input

def RecognizeSpeech(wavsignal, fs):
    data_input = GetFrequencyFeature3(wavsignal, fs)
    input_length = len(data_input)  #978
    input_length = input_length // 8  #122

    data_input = np.array(data_input, dtype=np.float32)

    data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)  #978,200,1
    batch_size = 1
    in_len = np.zeros((batch_size), dtype = np.int32)

    in_len[0] = input_length

    x_in = np.zeros((batch_size, 1600, AUDIO_FEATURE_LENGTH, 1), dtype=np.float32) #1,1600,200,1

    for i in range(batch_size):
        x_in[i, 0:len(data_input)] = data_input

    return x_in, in_len

def RecognizeSpeech_FromFile(filename):
    '''
    最终做语音识别用的函数，识别指定文件名的语音
    '''

    wavsignal,fs1 = read_wav_data(filename)  # 识别语音的特征 fs1=16000 len(wavsignal[0])=157000
    r, in_len = RecognizeSpeech(wavsignal, fs1)
    return r, in_len

