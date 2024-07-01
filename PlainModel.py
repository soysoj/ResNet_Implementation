# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

# conv_block: Conv2D & BatchNormalization structure
def conv_block(x, filters, kernel_size, strides):
    x = keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same' )(x)
    x = keras.layers.BatchNormalization()(x)
    return x

# Residual Block & Bottle Neck Block
def build_plain_block(input_layer, 
                    channel=64,
                    stride=1,
                    layer = 34
                    ):
     
    # 입력 레이어
    x = input_layer
    skip = input_layer

    # When # of layer is less than 50, the structure is
    # 50 레이어 미만이면 컨볼루션 구조가 2개로 구성된다.
    if layer < 50:
        # make the first conv
        x = conv_block(x,filters = channel, kernel_size = (3,3), strides = stride )
        # connect to the second conv
        x = keras.layers.Activation('relu')(x)
        # make the second conv
        x = conv_block(x, filters = channel, kernel_size = (3,3), strides = 1)
        x = keras.layers.Activation('relu')(x)
    
    # num_layer is larger than 50, the structure of residual and bottleneck block is different than the structure whose number of layer is less than 50.
    # 50 이상의 레이어를 가지면 컨볼루션블락이 3개로 구조가 변함. 
    #따라서 밑의 부분은 50 이상일때의 구조이다.
    else:
        # make the first conv block
        x = conv_block(x,filters = channel[0], kernel_size = (1,1), strides = stride)
        # connect to the second conv block
        x = keras.layers.Activation('relu')(x)
        # make the second conv block
        x = conv_block(x,filters = channel[1], kernel_size = (3,3), strides = 1)
        # connect to the third conv block
        x = keras.layers.Activation('relu')(x)
        # make the third conv block
        x = conv_block(x,filters = channel[2], kernel_size = (1,1), strides = 1)
        x = keras.layers.Activation('relu')(x)

    return x

def build_plain(input_shape=(128,128,3),
              num_block_list=[3,4,6,3],
              channel_list=[64,128,256,512],
              num_layer = 34,
              num_classes = 10):
    assert len(num_block_list) == len(channel_list) #모델을 만들기 전에 config list들이 같은 길이인지 확인합니다.

    input_layer = keras.layers.Input(shape=input_shape)
    x = conv_block(input_layer,filters = 64,kernel_size = (7,7), strides = 2)
    x = keras.layers.Activation('relu')(x)
    # MaxPool
    x = keras.layers.MaxPooling2D(pool_size=(3, 3),strides=2)(x)
    # Residual
    for i, (channel, num_block) in enumerate(zip(channel_list, num_block_list)):
        # residual block
        for j in range(num_block):
            x = build_plain_block(x, channel = channel, stride = 1, layer = num_layer)
            
    # Avg Pool
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes, activation = 'softmax')(x)

    model = keras.Model(inputs = input_layer, outputs = x)
    return model