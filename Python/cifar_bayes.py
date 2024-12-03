#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:36:35 2022

Using the cifar-10 data, classify the test set using both a naive bayes and 
non-naive bayes approach.  Reduce the images from 32x32x3 pixels into 1x1x3.  
Results:
    naive bayes: 0.1952 accuracy
    non-naive bayes: 0.2154 accuracy
    
Now use the non-naive bayes approach again, but this time don't reduce the 
images as much.  Keep the images square and see how the accuracy changes
based on the image size.
Results:
    2x2: 0.3051 accuracy
    4x4: 0.4032 accuracy
    8x8: 0.1271 accuracy
    16x16: covariance collapses
    

@author: Ash Batesole
"""


import time
start_time = time.time()
import matplotlib.pyplot as plt
import glob
import pickle
import numpy as np
import skimage.transform as transf
from scipy.stats import norm
from scipy.stats import multivariate_normal as multinorm

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


# convert image x(50000x32x32x3) to image y(50000x1x1x3)
# so converting from 32x32 pixels to 1x1
# so new image is only three values, the mean of each color channel
def cifar10_color(x):
    y = transf.resize(x,(1,1), preserve_range = True)
    return y


# naive bayes classifier
# computes the normal distribution parameters for a given class
# assuming features are independent
def cifar_10_naivebayes_learn(X,Y):
    
    # initialize arrays to store value of each color channel organized by class
    sorted_data_r = np.ones([10,n_tr])*-1
    sorted_data_g = np.ones([10,n_tr])*-1
    sorted_data_b = np.ones([10,n_tr])*-1
    
    
    # go through the training data and organize them by class and color
    for i in range(len(Y)):
        sorted_data_r[Y[i]][i] = X[i][0][0][0]
        sorted_data_g[Y[i]][i] = X[i][0][0][1]
        sorted_data_b[Y[i]][i] = X[i][0][0][2]
    

    # mask the empty elements in the arrays
    masked_data_r = np.ma.masked_equal(sorted_data_r, -1)
    masked_data_g = np.ma.masked_equal(sorted_data_g, -1)
    masked_data_b = np.ma.masked_equal(sorted_data_b, -1)

    
    # calculate mu, sigma, and prior p for each channel in each class
    for i in range(len(sorted_data_r)):
        naive_bayes_params_mu[i][0] = np.mean(masked_data_r[i])
        naive_bayes_params_mu[i][1] = np.mean(masked_data_g[i])
        naive_bayes_params_mu[i][2] = np.mean(masked_data_b[i])
        
        naive_bayes_params_sig[i][0] = np.std(masked_data_r[i])
        naive_bayes_params_sig[i][1] = np.std(masked_data_g[i])
        naive_bayes_params_sig[i][2] = np.std(masked_data_b[i])
        
        # all classes have equal probability in this dataset.  Since there are
        #   ten classes the prior probability for each class is 1/10
        naive_bayes_params_p[i] = 1/10
            
    return


# naives bayes classifier
# returns the optimal class for x
def cifar_10_classifier_naivebayes(x, mu, sig, p):
    pred = np.array([0]*len(x))
    
    # calculate the probabilities that x is from a given class and assign the 
    # class with the highest probability
    for i in range(len(x)):
        prob_temp = 0
        class_temp = 0
        # loop through the ten classes and compute the prob. of each one
        for j in range(0,10,1):
            # for naive bayes the total probability is the probabilities from each
            # color channel multiplied together
            prob = norm(mu[j][0],sig[j][0]).pdf(x[i][0][0][0])*\
                norm(mu[j][1],sig[j][1]).pdf(x[i][0][0][1])*\
                    norm(mu[j][2],sig[j][2]).pdf(x[i][0][0][2])*p[j][0]
            
            # update the current class and highest prob. 
            if prob > prob_temp:
                class_temp = j
                prob_temp = prob
        
        # add the predicted class into the array 
        pred[i] = class_temp
       
    return pred


# bayes classifier
# computes the normal distribution parameters for all ten classes,
# this time not assuming independence.  This will give us a 3D 
# norm. dist. instead of 3 x 1D norm. dist.
def cifar_10_bayes_learn(X,Y):
    
    # calculate dimensions of mu
    mu_size = X[0].shape
    
    # initialize an array to store the organized data
    # sorted_data1 = np.ones([10,n_tr,mu_size[0],mu_size[1],mu_size[2]])*-1 
    sorted_data2 = np.ones([10,n_tr,np.product(mu_size)])*-1 
    
    # go through the training data and organize them by class 
    for i in range(len(Y)):
        # sorted_data1[Y[i]][i] = X[i]
        sorted_data2[Y[i]][i] = X[i].flatten()
        
    # mask the empty elements 
    # masked_data1 = np.ma.masked_equal(sorted_data1, -1)
    masked_data2 = np.ma.masked_equal(sorted_data2, -1)
    
    # calculate mu, sigma, and prior p for each class
    for i in range(len(sorted_data2)):
        
        # for each class, mu is 3D, which is one mu per channel.  The data is
        # stored so that each column represents one channel, so we can just
        # find the mean of each channel (axis=0 uses columns)
        bayes_params_mu[i] = np.mean(masked_data2[i], axis = 0)
        
            
        # in the cov function it assumes each row is a separate variable by 
        # default, but ours is each column
        bayes_params_sig[i] = np.cov(masked_data2[i], rowvar=False)
        
        # all classes have equal probability in this dataset.  Since there are
        #   ten classes the prior probability for each class is 1/10
        bayes_params_p[i] = 1/10
    
    
    return


# bayes classifier
# returns the optimal class for x (using bayes instead of naive bayes)
def cifar_10_classifier_bayes(x, mu, sig, p):
    pred = np.array([0]*len(x))
    
    # calculate the probabilities that x is from a given class and assign the 
    # class with the highest probability
    for i in range(len(x)):
        prob_temp = 0
        class_temp = 0
        # loop through the ten classes and compute the prob. of each one
        for j in range(0,10,1):
            # for naive bayes the total probability is the probabilities from each
            # color channel multiplied together
            prob = multinorm(mu[j], sig[j], allow_singular=True).pdf(x[i].flatten())*p[j]

            
            # update the current class and highest prob. 
            if prob > prob_temp:
                class_temp = j
                prob_temp = prob
        
        # add the predicted class into the array 
        pred[i] = class_temp
    
    
    return pred


# convert image x(50000x32x32x3) to image (50000xpxpx3)
# so new image is pxp pixels for each color channel.
# this will return 3xpxp means for one image
def cifar10_2x2_color(x,p):
    y = transf.resize(x,(p,p), preserve_range = True)
    
    return y




# -------------------------------MAIN-----------------------------------------

# run the program using n samples
n_tr = 50000
n_te = 10000

# pull the data out of the cifar 10 download
# tr_data is the image data, tr_data_gt is the respective labels
tr_data = np.array([], dtype=np.uint8).reshape(0,3072)
tr_data_gt = np.array([], dtype=np.int64)

for filename in glob.glob('/home/mouse/Downloads/cifar-10-batches-py/data_batch_?'):
    datadict = unpickle(filename)
    tr_data = np.concatenate((tr_data, datadict["data"]), axis=0)
    tr_data_gt = np.concatenate((tr_data_gt, datadict["labels"]), axis=0)

# reshape X into 3072 into 32x32x3 
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


# resize images from 32x32x3 to 1x1x3
tr_data_3x1 = np.zeros([n_tr,1,1,3])
for i in range(len(tr_data_3x1)):
    tr_data_3x1[i] = cifar10_color(tr_data[i])
    
test_data_3x1 = np.zeros([n_te,1,1,3])
for i in range(len(test_data_3x1)):
    test_data_3x1[i] = cifar10_color(test_data[i])
    
    
    
    
# first we will make a classifier using naive bayes    
    
# compute mean and variance of each color channel for every class
# for example, for all images in class 1, find mean and var of each R, G, B
# first we will assume each color channel is an independent normal distribution

# create arrays to store mu, sigma, and prior p for each class
naive_bayes_params_mu = np.zeros([10,3])
naive_bayes_params_sig = np.zeros([10,3])
naive_bayes_params_p = np.zeros([10,1])

# learn the normal distribution parameters from the samples
cifar_10_naivebayes_learn(tr_data_3x1[0:n_tr], tr_data_gt[0:n_tr])

# now predict a class for an image using naive bayes
test_data_pred = cifar_10_classifier_naivebayes(test_data_3x1[0:n_te], naive_bayes_params_mu, 
                                naive_bayes_params_sig, naive_bayes_params_p)

print("Accuracy of naive bayes classifier:")
print(class_acc(test_data_pred, test_data_gt))
    



# this method has been commented out because it is also included in the next 
# section for part 3
# now we will make a classifier using non naive bayes

# do the same as above but this time using a multivariate normal distribution

# create arrays to store mu, sigma, and prior p for each class
bayes_params_mu = np.zeros([10,3])
bayes_params_sig = np.zeros([10,3,3])
bayes_params_p = np.zeros([10,1])

# # learn the 3D normal distribution parameters from the samples
# cifar_10_bayes_learn(tr_data_3x1[0:n_tr], tr_data_gt[0:n_tr])

# # predict a class for an image using non naive bayes
# test_data_pred = cifar_10_classifier_bayes(test_data_3x1[0:n_te], bayes_params_mu, 
#                                 bayes_params_sig, bayes_params_p)

# print("Accuracy of bayes classifier:")
# print(class_acc(test_data_pred, test_data_gt))





# now let's see how the classifier accuracies change with different sized images

# resize images from 32x32x3 to mxmx3
# covariance matrix collapses at 16x16, so stop before that size
accuracy = np.zeros(4)
m = np.array([1,2,4,8])
for k in range(0,4):
    
    # resize the images
    tr_data_mxm = np.zeros([n_tr,m[k],m[k],3])
    for i in range(len(tr_data_3x1)):
        tr_data_mxm[i] = cifar10_2x2_color(tr_data[i],m[k])
        
    test_data_mxm = np.zeros([n_te,m[k],m[k],3])
    for i in range(len(test_data_3x1)):
        test_data_mxm[i] = cifar10_2x2_color(test_data[i],m[k])
    
    # shape parameters to fit resized images
    bayes_params_mu = np.zeros([10,m[k]*m[k]*3])
    bayes_params_sig = np.zeros([10,m[k]*m[k]*3,m[k]*m[k]*3])
    
    # learn the 3D normal distribution parameters from the samples
    cifar_10_bayes_learn(tr_data_mxm[0:n_tr], tr_data_gt[0:n_tr])
    
    # predict a class for an image using non naive bayes
    test_data_pred = cifar_10_classifier_bayes(test_data_mxm[0:n_te], bayes_params_mu, 
                                    bayes_params_sig, bayes_params_p)
    
    # check accuracy
    accuracy[k] = class_acc(test_data_pred, test_data_gt)
    
    print(f'Accuracy of bayes classifier with {m[k]}x{m[k]} pixels:')
    print(accuracy[k])

# plot the accuracy results
plt.plot(m, accuracy, 'ro')
plt.axis([0,10,0,1])
plt.xlabel('m x m pixels')
plt.ylabel('accuracy')
plt.title('Accuracy vs image size using bayes classification')



# printing out time of program in seconds
print("program time in seconds:")
print(time.time() - start_time)
