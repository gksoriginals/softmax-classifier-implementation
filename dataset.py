#loading iris dataset from sklearn
from sklearn.datasets import load_iris

import numpy as np 
#for shuffling the dataset
def randomize(dataset,labels):
    permut=np.random.permutation(labels.shape[0])
    shuffled_features=dataset[permut,:]
    shuffled_labels=labels[permut]
    return shuffled_features,shuffled_labels
#one-hot encoder
def one_hot(data):
    x=np.array((data))
    labs=np.zeros((150,3))
    labs[np.arange(150),x]=1
    return labs
#fetch dataset from sklearn.dataset,divide into features and labels and one-hot encode labels for bilinear classification    
def fetch_data():
    data=load_iris()
    feat=data['data']
    lab=data['target']
    return feat,one_hot(lab)


    