# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:15:09 2019

@author: SHREYANK
"""
import numpy as np
from helper import formKernel

import scipy.io as sio
from sklearn import svm
def SVM(Xtrain, Ytrain, Xtest, Ytest, Xlearn, s, k):
    # Xtrain -> datapoints in training domain (typically domain A)
    # Xtest -> datapoints in testing domain (typically domain B)
    # Ytrain -> labels in training domain corresponding to datapoints of training domain
    # Ytest -> labels in test domain corresponding to datapoints of test domain
    # Xlearn -> training points
    # s -> learned kernel parameters

    # Features were combined together while learning S
    # optimised L matrix (represented as s)
    
    
    
    ##debug
#    Xtrain, Ytrain, Xtest, Ytest, Xlearn, s, k = X1,y1,X2,y2,Xlearn,S,1
    #
    if s is None:
        raise Exception('\'s\' is not defined');
    Ktrain_test  = formKernel(Xtrain, Xtest)
    Ktest_learn  = formKernel(Xtest, Xlearn)
    Ktrain_learn = formKernel(Xtrain, Xlearn)
    # @ is used for matrix multiplication
    Ktrain_test = Ktrain_test + Ktrain_learn  @ s @ Ktest_learn.T
    nKtest = np.ones((Xtest.shape[0],1), 'float64');
    nKtrain = np.ones((Xtrain.shape[0],1), 'float64');
    
    ###
    clf = svm.SVC(kernel=formKernel)
    clf.fit(Xtrain, Ytrain.T)
    
    predLabels = clf.predict(Xtest)
#    # call SVM
#    predLabels = kernelSVM(Ytrain, Ktrain_test.T, nKtrain, nKtest, k);
    # compute accuracy
    numRight = np.sum(predLabels == Ytest)
    acc  = numRight / len(predLabels);
    print ("Matching completed with accuracy of "+ str(acc*100) +" %")
    return acc;