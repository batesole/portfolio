#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 22:14:42 2022

Using the cifar-10 dataset, classify the images using a neural network.  The 
neural network must be full connected, takes an image as an input, and 
outputs a one-hot vector to assign a class.  Recommended to use keras or pytorch

Useful links:
    https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
        -basics of using keras, including an example
    https://keras.io/getting_started/intro_to_keras_for_engineers/
        -introduction to using keras
    https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
        -using keras with the cifar-10 dataset.  Includes the model that won a 
        performance competition in 2014
        

@author: Ash Batesole
"""

import time
start_time = time.time()
import matplotlib.pyplot as plt
import glob
import pickle
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from keras.optimizers import SGD



# function to unpickle the data from the cifar 10 data set
# unpickling converts the images from bytes to an object hierarchy
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


# function that computes the classification accuracy
def class_acc(pred, gt):
    accuracy = 0
    n = 0
    
    for i in range(len(pred)):
        n = n+1
        if pred[i]==gt[i]:
            accuracy = accuracy+1
    
    accuracy = accuracy/n   
    return accuracy



# -------------------------------MAIN-----------------------------------------

# run the program using n samples
n_tr = 5000
n_te = 1000



# pull the data out of the cifar 10 download
# tr_data is the image data, tr_data_gt is the respective labels
tr_data = np.array([], dtype=np.uint8).reshape(0,3072)
tr_data_gt = np.array([], dtype=np.int64)

for filename in glob.glob('/home/mouse/Downloads/cifar-10-batches-py/data_batch_?'):
    datadict = unpickle(filename)
    tr_data = np.concatenate((tr_data, datadict["data"]), axis=0)
    tr_data_gt = np.concatenate((tr_data_gt, datadict["labels"]), axis=0)

# leave data as a 1D shape
tr_data = tr_data.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

# pull out the label names 
labeldict = unpickle('/home/mouse/Downloads/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

# pull the test data out of cifar 10
datadict = unpickle('/home/mouse/Downloads/cifar-10-batches-py/test_batch')
test_data = datadict["data"]
test_data = test_data.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
test_data_gt = datadict["labels"]
test_data_gt = np.array(test_data_gt)

# neural networks process process vectorized & standardized representations
# so we need to standardize our images
# data must also be float32 to use keras
tr_data_norm = tr_data.astype('float32')
tr_data_norm = tr_data_norm/255.0
test_data_norm = test_data.astype('float32')
test_data_norm = test_data_norm/255.0

# convert ground truth vector to be "one hot" so that the one bit that is 1
# is the class of that data
tr_gt_bit = np.eye(10)[tr_data_gt[0:n_tr]]
te_gt_bit = np.eye(10)[test_data_gt[0:n_te]]
# could also use from keras.utils import to_categorical




# keras model
model = Sequential()

# input layer (we were told to use 5)
# activation: relu, as it is recommended over sigmoid for deep neural networks
#   sigmoid might be fine though since our network isn't very deep
# kernel_init: he_uniform initializes weights using a uniform distribution that
#   is based on the number of inputs
model.add(Dense(5, input_shape=(32,32,3), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(32, 5, activation="relu", kernel_initializer='he_uniform'))
# model must be flattened to 1D to match output type of 1D
model.add(Flatten())

# output layer must be ten output sigmoid
model.add(Dense(10, activation='sigmoid'))

# compile the model (adam is a popular version of gradient descent)
opt = SGD(learning_rate=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# loss: use categorical_crossentropy for multi categorical classification
# optimizer: using a recommended one for cifar-10
# metrics: use accuracy because that is what we are monitoring


# fit the model
tr_hist = model.fit(tr_data_norm[0:n_tr], tr_gt_bit, epochs=20, 
                    validation_data=(test_data_norm[0:n_te], te_gt_bit), verbose=0)
# evaluate accuracy
acc = model.evaluate(test_data_norm[0:n_te], te_gt_bit, verbose=0)

# plot loss
plt.subplot(211)
plt.title('Cross Entropy Loss')
plt.plot(tr_hist.history['loss'], color='blue', label='train')
plt.plot(tr_hist.history['val_loss'], color='orange', label='test')

# plot accuracy
plt.subplot(212)
plt.title('Classification Accuracy(training data)')
plt.plot(tr_hist.history['accuracy'], color='blue', label='train')
plt.plot(tr_hist.history['val_accuracy'], color='orange', label='test')
plt.tight_layout(pad=1.0)

print('accuracy: %.3f'  % (acc[1] * 100.0))
print('program time in seconds: %.3f' % (time.time() - start_time))
