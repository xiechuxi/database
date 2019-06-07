#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:59:05 2019

@author: xiechuxi
"""
from comet_ml import Experiment
import tensorflow as tf



session_conf = tf.ConfigProto()  # Use all CPU cores

from skimage.io import imread
from keras import layers
from keras import layers, models, optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Reshape, Layer, Lambda
import tensorflow as tf
from keras import backend as K
from keras import initializers
from sklearn import preprocessing
from keras.layers import * 
from keras.models import Model
from keras import callbacks
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
import os




"""
initialize of the cardinality
"""

cardinality=8
img_height= 224
img_width= 224
img_channels= 3

"""
you can add LeakyReLU() or change it to regular relu or exponential relu
"""


def common_layers(y):
    
    y=layers.BatchNormalization()(y)
    y=layers.LeakyReLU(alpha=0.2)(y)
    
    return y


def convolutionarylayers_group(y, n_channels, _strides):
    if cardinality==1:
        return layers.Conv2D(n_channels, kernel_size=(3,3),strides=_strides, padding='same')(y)
    
    assert not n_channels % cardinality
    
    _d=n_channels // cardinality
    
    groups=[]
    
    for j in range(cardinality):
        group=layers.Lambda(lambda z: z[:,:,:,j*_d:j*_d+_d])(y)
        m=layers.Conv2D(_d, kernel_size=(3,3),strides=_strides, padding='same')(group)
        groups.append(m)
        
        
    
    y=layers.concatenate(groups)
    
    
    
    return y


def residual_block(y, n_channels_in,n_channels_out,_strides=(1,1),project_shortcut=False):
    
    shortcut=y
    """bottle neck layer"""
    y= layers.Conv2D(n_channels_in,kernel_size=(1,1),strides=(1,1),padding='same')(y)
    y= common_layers(y)
    """ResNeXt if cardinality=1 this is Resnet"""
    y= convolutionarylayers_group(y,n_channels_in,_strides=_strides)
    y= common_layers(y)
    y= layers.Conv2D(n_channels_out,kernel_size=(1,1),strides=(1,1),padding='same')(y)
    y= layers.BatchNormalization()(y)
    
    if project_shortcut or _strides != (1, 1):
        #when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        #when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(n_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
    
    """add the input with output"""
    
    y=layers.add([shortcut,y])
    y=layers.LeakyReLU(alpha=0.2)(y)
    
    
    return y

def build_model(x):
    
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = common_layers(x)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    """continously add multiple layers"""
    for i in range(3):
        project_shortcut = True if i == 0 else False
        x = residual_block(x, 128, 256, project_shortcut=project_shortcut)
        
    for i in range(4):
        """ down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2"""
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 256, 512, _strides=strides)
        
    for i in range(6):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 512, 1024, _strides=strides)
        
    for i in range(3):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 1024, 2048, _strides=strides)
        
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(14)(x)
    
    return x



    
def preparedataframe(dataframeip,split_percent):
    df=pd.read_csv(dataframeip)
    df=df[df['specie']!='NEGATIVES']
    df=df.reindex(np.random.permutation(df.index))
    split_number=int(len(df['specie'])*(1-split_percent))
    traindf=df.iloc[:, :split_number]
    validatedf=df.iloc[:, split_number:]
    return traindf,validatedf


def loaddata_train(dataframeip,batchsize,split_percent,epnumber):
    
    df=pd.read_csv(dataframeip)
    
    train_data_gen_args = dict(
            rotation_range=360.0,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect',
            rescale=1./255
            )
    
    
    traingen= ImageDataGenerator(train_data_gen_args)
    
    train_generator = traingen.flow_from_dataframe(
            dataframe=df,
            x_col="filename",
            y_col="specie",
            target_size=(224, 224),
            class_mode="categorical",
            shuffle=True,
            batch_size=batchsize,
            validation_split=split_percent
            )
   
     
    """define the model"""
    image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    network_output = build_model(image_tensor)
    model = models.Model(inputs=[image_tensor], outputs=[network_output])
    
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    
    
    model.fit_generator(train_generator,
                        steps_per_epoch=df.shape[0] // batchsize + 1,
                        epochs=epnumber
                        )
    
    
    return model


if __name__ == "__main__":
    loaddata_train('/Users/xiechuxi/Desktop/EO_rank_data.csv',128,0.2,50)
    
    
    
    
    
    
    
    



        
    
    
    