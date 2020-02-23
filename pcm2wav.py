# coding=utf-8
import wave

#pcm_path = r'speech_voice/01.pcm'

def pcm2wav(pcm_path):
    # 打开并去读pcm音频
    pcmfile = open(pcm_path, 'rb')
    pcmdata = pcmfile.read()
    pcmfile.close()

    # 设置wav 音频参数
    channels = 2
    bits = 16
    sample_rate = 16000

    # 定义wav音频的生成路径和名称
    wave_path_front = pcm_path[:-4]
    wave_path = wave_path_front + '.wav'

    # 创建wav音频文件
    wavfile = wave.open(wave_path, 'wb')

    wavfile.setnchannels(channels)
    wavfile.setsampwidth(bits // 8)
    wavfile.setframerate(sample_rate)

    # 写入wav音频数据
    wavfile.writeframes(pcmdata)

    wavfile.close()

    return wave_path


#print pcm2wav(pcm_path)
