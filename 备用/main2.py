# coding=utf-8
import hiai
from hiai.nn_tensor_lib import DataType
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
print sys.getdefaultencoding()
import os
import numpy as np
import time
import ctc_func as K
import pcm2wav as L
from pcm2wav import pcm2wav
from get_features import RecognizeSpeech_FromFile
from make_input_tensor import make_input_tensor

from get_symbol_list import GetSymbolList


from language_model_func import ModelLanguage


# 定义文件获取路径
speech_recog_model = 'speech_recog_model/test_model.om' # 语音识别的声学模型
speech_voice_path = '01.pcm' # 开发板保存的语音路径
speech_voice_path2 = 'speech_voice/chang.wav'


def GetDataSet(speech_voice_path):
    # 将pcm数据转换为wav
    wave_path = L.pcm2wav(speech_voice_path) # 已完成

    # 读取wav音频特征
    features, in_len = RecognizeSpeech_FromFile(wave_path) # 已完成
    #print features

    # 将wav音频特征转换为模型输入向量
    input_tensor = make_input_tensor(features) # 已完成

    return input_tensor, in_len

def GetDataSet2(speech_voice_path):
    features, in_len = RecognizeSpeech_FromFile(speech_voice_path) #1,1600,200,1  in_len=122 全0矩阵
    features1=np.reshape(features,[1,1600,200,1])
    '''features1=np.swapaxes(features,3,1)
    features1=np.swapaxes(features1,3,2)'''

    features1=np.transpose(features1,(0,3,1,2)).copy()
    #print(features1.shape)
    #print("tensor输入张量")
    input_tensor = make_input_tensor(features1)
    return input_tensor, in_len


def CreateGraph(model):

    # 调用get_default_graph获取默认Graph，再进行流程编排
    myGraph = hiai.hiai._global_default_graph_stack.get_default_graph()

    if myGraph is None:
        print 'Get default graph failed'
        return None

    nntensorList = hiai.NNTensorList()

    # 不实用DVPP缩放图像，使用opencv缩放图片
    resultInference = hiai.inference(nntensorList, model, None) # 不确定其功能

    if (hiai.HiaiPythonStatust.HIAI_PYTHON_OK == myGraph.create_graph()):
        print 'create graph ok !'
        return myGraph
    else:
        print 'create graph failed, please check log.'
        return None


def GraphInference(graphHandle, inputTensorList):
    if not isinstance(graphHandle, hiai.Graph):
        print 'graphHandle is not Graph object'
        return None

    resultList = graphHandle.proc(inputTensorList)
    return resultList


def SpeechPostProcess(resultList, in_len): # 未完成！！！
    #print("--------------------")
    #print(resultList)
    resultList1 = resultList[0]
    #print "resultList[0] shape" + str(resultList1.shape)  # (200,1,1,1424)
    #print resultList1[0][0][0][1423]
    # *****************
    resultArray = resultList[0]
    batchNum = resultArray.shape[0]  #200
    #print "batchNum: " + str(batchNum)
    confidenceNum = resultArray.shape[1]  #1
    #print "confidenceNum: " + str(confidenceNum)
    #print "resultArray: " + str(resultArray)
    #print "resultArray shape: " + str(resultArray.shape)  # (200,1,1,1424)
    confidenceList = resultArray[:, 0, 0, :]
    resultArray1=np.swapaxes(resultArray,0,2)
    resultArray2 = np.swapaxes(resultArray1, 0, 1)
    confidenceList = resultArray2[0]
    #print(confidenceList)
    #print("+++++++++++++++++++++++++")
    #print "confidenceList shape: " + str(confidenceList.shape)  # (1,200,1424)

    confidenceArray = np.array(confidenceList)
    #print "confidenceArray: " + str(confidenceArray)
    resultList = confidenceArray
    #resultList = resultList[:, :, :]
    #print("----------------------")
    #print(resultList.shape)

    ret = K.ctc_decode(resultList, in_len, greedy = True, beam_width=100, top_paths=1)
    #print(ret[0])
    ret1 = K.get_value(ret[0][0])
    #print ret1
    ret1 = ret1[0]

    list_symbol_dic = GetSymbolList()

    r_str = []
    for i in ret1:
        r_str.append(list_symbol_dic[i])


    print "拼音序列识别结果：" + str(r_str)
    ml = ModelLanguage('language_model')

    ml.LoadModel()

    #ml = LoadModel()

    str_pinyin = r_str
    #str_pinyin = ['ni3', 'hao3', 'a1']
    #print str_pinyin
    r = ml.SpeechToText(str_pinyin)

    return r


def main():

    print "Start get data set"
    Input_tensor, in_len = GetDataSet(speech_voice_path)

    # 判断Input_data是否正确获取
    if (Input_tensor == None):
        print 'Get input data failed'

    # 加载语音识别的声学模型
    print "Start load speech model"
    inferenceModel = hiai.AIModelDescription('asr', speech_recog_model)
    if inferenceModel is None:
        print 'Load model failed'
        return None


    # 初始化Graph
    print "Start init Graph"
    myGraph = CreateGraph(inferenceModel)

    # 开始模型推理
    print "Start inference"
    resultList = GraphInference(myGraph, Input_tensor)

    list_shape = np.array(resultList).shape
    #print(list_shape)
    #print("++++++++++++++++++++++++++++++")
    #print resultList[0]

    if resultList is None:
        print "Inference failed"

    # 对结果进行后处理


    final_result = SpeechPostProcess(resultList,in_len)

    print '文本识别结果： ' + str(final_result)

    hiai.hiai._global_default_graph_stack.get_default_graph().destroy()

    print 'Speech Recognizition Finished !'


if __name__ == "__main__":
    main()