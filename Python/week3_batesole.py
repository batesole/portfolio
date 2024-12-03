#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 20:17:27 2022

This program is about learning and applying k nearest neighbor to cifar 10.
A function that calculates the class prediction accuracy is first tested
using the ground truth as the predictors, then secondly tested by randomly 
assigning classes to a predictor.  Then we use 1 nearest neighbor (1NN) to 
predict a class.  During testing a prediction accuracy of 0.2-0.35 was obtained
using 1NN, which was calculated using euclidian distance.

Cifar 10 provides 50,000 images assigned to 10 classes for training data,
plus another 10,000 images for testing data.


@author: Ash Batesole
"""

import time
start_time = time.time()
import glob
import copy
import pickle
import numpy as np
import random

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


# function that randomly assigns labels
def cifar10_classifier_random(x):
    label = random.randint(0,9)   
    return label


# function that assigns labels using 1 nearest neighbor
# x is a vector of data to be labeled
def cifar10_classifier_1nn(x, trdata, trlabels):
    label = 0
    dist = np.sum((x - trdata[0])^2)
    temp_dist = 0
    
    for i in range(len(trdata)):
        temp_dist = np.sum((x - trdata[i])^2)
        
        if temp_dist < dist:
            dist = temp_dist
            label = trlabels[i]
    
    return label




# ------MAIN---------


# pull the data out of the cifar 10 download
# X is the image data, Y is the respective labels
X = np.array([], dtype=np.uint8).reshape(0,3072)
Y = np.array([], dtype=np.int64)

for filename in glob.glob('/home/mouse/Downloads/cifar-10-batches-py/data_batch_?'):
    datadict = unpickle(filename)
    X = np.concatenate((X, datadict["data"]), axis=0)
    Y = np.concatenate((Y, datadict["labels"]), axis=0)

# reshape X into 3072 into 3x32x32 
X = X.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

# pull out the label names 
labeldict = unpickle('/home/mouse/Downloads/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]



# test the classification accuracy function 
# first use the ground truth as the predictor
testpred = copy.deepcopy(Y)
testgt = Y
print(class_acc(testpred, testgt))
#returns 1.0, which is good, since the predictor was the ground truth


# test the classification accuracy using randomly assigned labels
for i in range(len(testpred)):
    testpred[i] = cifar10_classifier_random(X[i])

print(class_acc(testpred, testgt))
# returns values around 0.1, which is correct.  There are ten classes,
# so if they are assigned randomly the accuracy should be ~1/10



# use 1 nearest neighbor to assign classes to the test data
# pull the test data out of cifar 10
datadict = unpickle('/home/mouse/Downloads/cifar-10-batches-py/test_batch')
test_data = datadict["data"]
test_data = test_data.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
test_data_gt = datadict["labels"]
test_data_gt = np.array(test_data_gt)

# predict labels for the test data using 1NN
pred = np.array([0]*10000)
for i in range(len(pred)):
    pred[i] = cifar10_classifier_1nn(test_data[i], X, Y)

# check how accurate those labels were
print(class_acc(pred, test_data_gt))
# during testing accuracy was between 0.2 - 0.35

# printing out time of program in seconds
print(time.time() - start_time)
