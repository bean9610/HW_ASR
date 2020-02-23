# coding=utf-8
# please import your module
import hiai
from hiai.nn_tensor_lib import DataType
#import numpy as np

def make_input_tensor(wav_features):

    input_tensor_height = wav_features.shape[2]
    input_tensor_width = wav_features.shape[3]

    #wav_features = np.reshape(wav_features, (0,3,1,2))
    #print 555
    #print wav_features.shape[0]
    #print wav_features.shape[1]
    #print wav_features.shape[2]
    #print wav_features.shape[3]

    #print wav_features.shape
    input_tensor = hiai.NNTensor(wav_features, input_tensor_height, input_tensor_width, 1, "wav_features", DataType.FLOAT32_T, wav_features.size)
    width0 = input_tensor.width
    height0 = input_tensor.height
    channel0 = input_tensor.channel
    #print str(height0)
    #print str(width0)
    #print str(channel0)

    nntensorList = hiai.NNTensorList(input_tensor)

    return nntensorList
    #return wav_features