# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:35:56 2020

@author: yvonn
"""
import numpy as np

x = np.array([1,2,1,2,2,1,1,2])

np.random.seed(1)
idx = np.where(x==1)[0]
print(idx)

y = np.random.choice(idx)
print(y)
y = np.random.choice(idx)
print(y)

import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical


 #Load MNISt dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Check number of samples (60000 in training and 10000 in test)
# Each image has 28 x 28 pixels
print("Train Image Shape: ", train_images.shape, "Train Label Shape: ", train_labels.shape) 
print("Test Image Shape: ", test_images.shape, "Test Label Shape: ", test_labels.shape) 

#  Visualizing a random image (11th) from training dataset
print("Visualizing a random image (11th) from training dataset")
_ = plt.imshow(train_images[0])
print(type(train_images))
print(train_labels)

# Preprocessing: Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

import warnings
warnings.filterwarnings("ignore")  # Ignore some warning logs


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


#  Define a Feed-Forward Model with 2 hidden layers with dimensions 392 and 196 Neurons
model = Sequential([
  Dense(784, activation='relu', input_shape=(28*28,), name="first_hidden_layer"),
  Dense(784//2, activation='relu', name="second_hidden_layer"), Dropout(0.25),
  Dense(10, activation='softmax'),
])

#  Validate your Model Architecture
print(model.summary())

# Compile model
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'],)

# Flatten the images into vectors (1D) for feed forward network
flatten_train_images = train_images.reshape((-1, 28*28))
flatten_test_images = test_images.reshape((-1, 28*28))
print(train_labels)

# Train model
model.fit(flatten_train_images, to_categorical(train_labels), epochs=10, batch_size=256,)

import matplotlib.image as img

image = img.imread('C:/Users/yvonn/Downloads/Wallpaper/gokyo ri.jpg')


from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D


# =============================================================================
# def optimize_cnn(hyperparameter):
#   
#   # Define model using hyperparameters 
#   cnn_model = Sequential([Conv2D(32, kernel_size=hyperparameter['conv_kernel_size'], activation='relu', input_shape=(28,28,1)), 
#             Conv2D(32, kernel_size=hyperparameter['conv_kernel_size'], activation='relu'), 
#             MaxPooling2D(pool_size=(2,2)), Dropout(hyperparameter['dropout_prob']),
#             Conv2D(64, kernel_size=hyperparameter['conv_kernel_size'], activation='relu'),
#             Conv2D(64, kernel_size=hyperparameter['conv_kernel_size'], activation='relu'), 
#             MaxPooling2D(pool_size=(2,2)), Dropout(hyperparameter['dropout_prob']), 
#             Flatten(),
#             Dense(512, activation='relu'), 
#             Dense(10, activation='softmax'),])
#   
#   cnn_model.compile(optimizer=hyperparameter['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'],)
# 
#   # create a training (50K samples) and validation (10K samples) subsets from training images.
#   # Validation subset will be used to find the optimal hyperparameters
# =============================================================================
  
# =============================================================================
# act = 'relu'
# a = [0,1,3]
# layers = [
#         Dense(2304, activation = act, input_shape=(48*48,), name="1st_hidden_layer"),
#         Dense(2304//2, activation = act, name="2nd_hidden_layer"),
#         Dense(2304//4, activation = act, name="3rd_hidden_layer"),
#         Dense(2304//8, activation = act, name="4th_hidden_layer"),
#         Dense(2304//16, activation = act, name="5th_hidden_layer"),
#         Dense(2304//32, activation = act, name="6th_hidden_layer"),
#         Dense(7, activation = 'softmax'),]
# x = [layers[i] for i in a]+[layers[6]]
# mod = Sequential(x)
# print(mod.summary())
# =============================================================================

# =============================================================================
# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()
# =============================================================================

from skimage.measure import block_reduce
train_images = block_reduce(train_images,block_size=(1,2,2),func=np.mean)
plt.imshow(train_images[0])

