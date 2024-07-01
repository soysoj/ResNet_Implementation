# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

# conv_block: Conv2D & BatchNormalization structure
def conv_block(x, filters, kernel_size, strides):
    x = keras.layers.Conv2D(filter = filters, kernel_size = kernel_size, strides = strides, padding = 'same' )(x)
    x = keras.layers.BatchNormalization()(x)
    return x

# Residual Block & Bottle Neck Block
def build_plain_block(input_layer, 
                    channel=64,
                    stride=1,
                    layer = 34
                    ):    
    # stride = 1로 들어오면 normal block
    # stride = 2로 들어오면 bottle neck   
     
    # 입력 레이어
    x = input_layer
    skip = input_layer

    # When # of layer is less than 50, the structure is
    # 50 레이어 미만이면 컨볼루션 구조가 2개로 구성된다.
    if layer < 50:
        # make the first conv
        x = conv_block(filters = channel, kernel_size = (3,3), stride = stride ,padding = 'same')
        # connect to the second conv
        x = keras.layers.Activation('relu')(x)
        # make the second conv
        x = conv_block(filters = channel, kernel_size = (3,3), stride = 1, padding = 'same')
            
        # skip connection이 없는 구조여야하기때문에 바로 relu로
        x = keras.layers.Activation('relu')
    
    # num_layer is larger than 50, the structure of residual and bottleneck block is different than the structure whose number of layer is less than 50.
    # 50 이상의 레이어를 가지면 잔차블락과 병목블락에서 컨볼루션블락이 3개로 구조가 변함. 
    #따라서 밑의 부분은 50 이상일때의 구조이다.
    else:
        # make the first conv block
        x = conv_block(filters = channel[0], kernel_size = (1,1), stride = stride, padding = 'same')(x)
        # connect to the second conv block
        x = keras.layers.Activation('relu')
        # make the second conv block
        x = conv_block(filters = channel[1], kernel_size = (3,3), stride = 1, padding = 'same' )
        # connect to the third conv block
        x = keras.layers.Activation('relu')
        # make the third conv block
        x = conv_block(filters = channel[2], kernel_size = (1,1), stride = 1, padding = 'same')
        
        x = keras.layers.Activation('relu')

    return x

def build_ResNet(input_shape=(32,32,3),
              num_block_list=[2,2,3,3,3],
              channel_list=[64,128,256,512,512],
              num_layer = 34,
              num_classes = 10):
    assert len(num_block_list) == len(channel_list) #모델을 만들기 전에 config list들이 같은 길이인지 확인합니다.

    input_layer = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(64, (7, 7), strides=2, padding='same')(input_layer)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    # MaxPool
    x = keras.layers.MaxPooling2D(pool_size=(3, 3),strides=2)(x)
    # Residual
    for i, (channel, num_block) in enumerate(channel_list, num_block_list ):
        # bottle neck block
        if i != 1:
            x = build_plain_block(x, filters = channel, stride = 2, layer = num_layer)
        # residual block
        for _ in range(num_block):
            x = build_plain_block(x, filters = channel, stride = 2, layer = num_layer)
                   
    # Avg Pool
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes, activation = 'softmax')(x)

    model = keras.Model(inputs = input_layer, output = x)
    return model