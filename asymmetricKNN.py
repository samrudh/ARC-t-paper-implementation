### Perform KNN
'''
Performs KNN based classification on asymmetric test and training data

'''
import numpy as np
from helper import formKernel


def kernelKNN (Ytrain, Ktrain_test, nKtrain, nKtest, k):

    # compute distance
    (rows, cols) = Ktrain_test.shape
    distMatrix = np.zeros((rows, cols), 'float64');
    for i in range(0,rows):
        for j in range(0,cols):
            distMatrix[i,j] = nKtest[i] + nKtrain[j] - 2 * Ktrain_test[i, j];

    indices = np.argsort(distMatrix, axis=1);
    preds = np.zeros(rows,"int")

    for i in range(0, rows):
        counts = np.zeros(32);
        for j in range(0, k):
            if Ytrain[indices[i,j]] > np.count_nonzero(counts):
                counts[Ytrain[indices[i,j]]] = 1;
            else:
                counts[Ytrain[indices[i,j]]] = counts[Ytrain[indices[i,j]]] + 1;
        preds[i] = np.argmax(counts)
    return preds

def asymmetricKNN(Xtrain, Ytrain, Xtest, Ytest, Xlearn, s, k):
    # Xtrain -> datapoints in training domain (typically domain A)
    # Xtest -> datapoints in testing domain (typically domain B)
    # Ytrain -> labels in training domain corresponding to datapoints of training domain
    # Ytest -> labels in test domain corresponding to datapoints of test domain
    # Xlearn -> training points
    # s -> learned kernel parameters

    # Features were combined together while learning S
    # optimised L matrix (represented as s)
    if s is None:
        raise Exception('\'s\' is not defined');
    Ktrain_test  = formKernel(Xtrain, Xtest)
    Ktest_learn  = formKernel(Xtest, Xlearn)
    Ktrain_learn = formKernel(Xtrain, Xlearn)
    # @ is used for matrix multiplication
    Ktrain_test = Ktrain_test + np.matmul(np.matmul(Ktrain_learn, s), Ktest_learn.T)
    nKtest = np.ones(Xtest.shape[0], 'int');
    nKtrain = np.ones(Xtrain.shape[0], 'int');

    # call knn
    predLabels = kernelKNN(Ytrain, Ktrain_test.T, nKtrain, nKtest, k);
    # compute accuracy
    numRight = np.sum(predLabels == Ytest)
    acc = numRight / len(predLabels);
    print ("Asymmetric Matching completed with accuracy of " + str(acc*100) +" %")
    return acc;
