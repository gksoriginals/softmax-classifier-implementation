# -*- coding: utf-8 -*-



import numpy as np
"""
This is a naive implementaton of softmax,cross entropy loss function with numpy and python.
Gradient descent is used for optimization
"""
def softmax_cross_entropy_error(feat,lab,weight,bias):
    #calculate the softmax output,cross entropy error
    cross_entr=0
    
    for fea in range(0,len(feat)):
        #softmax =e**z/sum(e**z)
        #where z=w*x+b 
        #exponent of z is calculated(unnormalized) and normalized it by dividing wth sum of e**z
        softma=np.exp(np.matmul(feat[fea],weight)+bias)/np.sum(np.exp(np.matmul(feat[fea],weight)+bias))
        #cross entropy function multiply target with log of guessed output ie softmax output
            
        cross_entr += -(1/len(feat))*lab[fea].dot(np.log(softma))
        if(fea%50==0):
            print("guess=",softma)
            print("actual=",lab[fea])
    return cross_entr*100
def gradient_find(current_w,current_b,feat,lab,l_rate):
    b_grad=0
    w_grad=0
    N=float(len(feat))
   
    for fea in range(0,len(feat)):
        softma=np.exp(np.matmul(feat[fea],current_w)+current_b)/np.sum(np.exp(np.matmul(feat[fea],current_w)+current_b))
        #calculating gradient,the slope of the function so we can follow in its opposite direction to find the minima       
        b_grad+=(1/N)*np.subtract(softma,lab[fea])
        w_grad+=(1/N)*np.matmul(feat[fea].reshape(1,4).T,(softma-lab[fea]).reshape(1,3))
    new_b=current_b-l_rate*b_grad#following the slope to find the bias and weights 
    new_w=current_w-l_rate*w_grad#with minimum value of error
    return new_w,new_b
def gradient_descent_runner(feat,lab,first_w,first_b,l_rate,num_iter):
    #function for running gradient descent for a given number ooof iteration with learning rate given    
    b=first_b
    w=first_w
    for i in range(num_iter):
        w,b=gradient_find(w,b,feat,lab,l_rate)
    return w,b

    
        
    

    
    
    
    
    