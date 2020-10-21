# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 10:54:43 2020

@author: yvonn
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

import tensorflow

from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore")  # Ignore some warning logs

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D

from skimage.measure import block_reduce


def importFile(filename):
    datafile = csv.reader(open(filename,'r'))
    data = []
    label = []
    
    for idx, row in enumerate(datafile):
        if idx == 0:
            continue
        else:
            data.append(row[1].split(' '))
            label.append(float(row[0]))
            
    for idx, line in enumerate(data):
        data[idx] = [float(ele) for ele in line]
        
    return np.array(data), np.array(label)

# Visualize 7 random images from training set for each emotion
def displayImages(data,label,savefig):
    imgIdx = []
    
    np.random.seed(0)
    for i in range(7):
        idx = np.where(label==i)[0] # Find indices of images that belong to class(emotion) i
        imgIdx.append(np.random.choice(idx)) # Randomly select one of the images and save the index
    imgIdx = np.array(imgIdx).flatten()
    
    plt.figure(figsize=(6*3,7*3))
    for i in range(7):
        plt.subplot(3,3,i+1)
        plt.imshow(data[imgIdx[i]].reshape(48,48))
        plt.title("Emotion {}".format(i))
    #plt.suptitle("Three random samples from training set")
    
    if savefig == True:
        plt.savefig("Random samples.png",dpi=160)
        
# FNN
def FNN(nlayer,act,dropout,train,trainLabel,epoch):
    inputSize = 24*24
    layers = [
        Dense(inputSize, activation = act, input_shape=(inputSize,), name="1st_hidden_layer"),Dropout(dropout),#0,1
        Dense(inputSize//2, activation = act, name="2nd_hidden_layer"),Dropout(dropout),#2,3
        Dense(inputSize//4, activation = act, name="3rd_hidden_layer"),Dropout(dropout),#4,5
        Dense(inputSize//8, activation = act, name="4th_hidden_layer"),Dropout(dropout),#6,7
        Dense(inputSize//16, activation = act, name="5th_hidden_layer"),Dropout(dropout),#8,9
        Dense(inputSize//32, activation = act, name="6th_hidden_layer"),Dropout(dropout),#10,11
        Dense(7, activation = 'softmax'),]
    
    model = Sequential([layers[i] for i in nlayer]+[layers[12]])
    print(model.summary())
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'],)
    
    train = np.array(train).reshape((-1,24*24))
    model.fit(train, to_categorical(trainLabel), epochs=epoch, batch_size=128)
    
    return model
    

'''
    main program
'''
data,label = importFile('D:\Github\CSCE-633\Homeworks\Homework3\Q2_Train_Data.csv')
dataVal, labelVal = importFile('D:\Github\CSCE-633\Homeworks\Homework3\Q2_Validation_Data.csv')
#dataTest, labelTest = importFile('D:\Github\CSCE-633\Homeworks\Homework3\Q2_Test_Data.csv')

# Downsample data to dimension 24*24 for faster processing
data = block_reduce(data.reshape(-1,48,48),block_size=(1,2,2),func=np.mean)
#plt.imshow(data[0])
dataVal = block_reduce(dataVal.reshape(-1,48,48),block_size=(1,2,2),func=np.mean)
#dataTest = block_reduce(dataTest.reshape(-1,48,48),block_size=(1,2,2),func=np.mean)

data = tensorflow.image.per_image_standardization(data.reshape(28709,24,24,1))
dataVal = tensorflow.image.per_image_standardization(dataVal.reshape(3589,24,24,1))
#dataTest = tensorflow.image.per_image_standardization(dataTest.reshape(3589,48,48,1))

#displayImages(data,label,True)
modelFNN = FNN([0,1,2,3],'relu',0.5,data,label,30)
modelFNN.evaluate(np.array(dataVal).reshape((-1,24*24)),to_categorical(labelVal))




    