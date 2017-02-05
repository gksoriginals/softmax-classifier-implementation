# -*- coding: utf-8 -*-

#importing classifier containing softmax function,cross entropy and gradient_descent
from classifier import softmax_cross_entropy_error,gradient_descent_runner
from dataset import fetch_data,randomize
import numpy as np
from numpy import random

def run():
    #load features (150,4) and labels (150,3)
    features,labels=fetch_data()
    #shuffling features and labels
    features,labels=randomize(features,labels)
    random.seed(0)
    #initializing weights with random values
    weights=np.random.random((4,3))
    #initializing biases with random values
    biases=np.random.random((3))
    #number of times gradient descent to be iterated     
    num_iter=10000
    #learning rate 
    l_rate=0.005
    print("b=",biases) #initial biases
    print("w=",weights) #initial weights
    #calculating initial error
    print("error={0}%".format(softmax_cross_entropy_error(features,labels,weights,biases)))
    print("Running classifier....")
    #Running gradient descent
    weights,biases=gradient_descent_runner(features,labels,weights,biases,l_rate,num_iter)
    print("b=",biases)#biases values after running classifier
    print("w=",weights)#weights values after running classifier
    print("error={0}%".format(softmax_cross_entropy_error(features,labels,weights,biases)))
    #error after running the program 
if __name__=='__main__':
    run()


###########################################OUTPUT##############################################

"""
b= [ 0.56804456  0.92559664  0.07103606]
w= [[ 0.5488135   0.71518937  0.60276338]
 [ 0.54488318  0.4236548   0.64589411]
 [ 0.43758721  0.891773    0.96366276]
 [ 0.38344152  0.79172504  0.52889492]]
guess= [ 0.11491955  0.58894483  0.29613562]
actual= [ 1.  0.  0.]
guess= [ 0.02417515  0.68201728  0.29380758]
actual= [ 0.  1.  0.]
guess= [ 0.01047284  0.73688261  0.25264455]
actual= [ 0.  0.  1.]
error=127.57215588040505%


Running classifier....


b= [ 0.89518099  1.27909495 -0.60959868]
w= [[ 1.32442085  1.15762669 -0.61528129]
 [ 2.08761584  0.19311681 -0.66630056]
 [-1.45854048  0.69743801  3.05412545]
 [-0.50847225 -0.08447748  2.2970112 ]]
guess= [  9.30802089e-01   6.91900641e-02   7.84694703e-06]
actual= [ 1.  0.  0.]
guess= [ 0.03034742  0.83405964  0.13559294]
actual= [ 0.  1.  0.]
guess= [ 0.0009035   0.26344645  0.73565006]
actual= [ 0.  0.  1.]
error=17.27467814880828%
"""    