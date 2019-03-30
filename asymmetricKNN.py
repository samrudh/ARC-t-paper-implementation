### Perform KNN
'''
Performs KNN based classification on asymmetric test and training data

'''
import numpy as np
from helper import formKernel


def kernelKNN (Ytrain, Ktrain_test, nKtrain, nKtest, k):
    Ktrain_test = Ktrain_test.T;

    # compute distance
    (rows, cols) = Ktrain_test.shape
    distMatrix = np.zeros((rows, cols), 'float64');
    for i in range(0,rows-1):
        for j in range(0,cols-1):
            distMatrix[i,j] = nKtest[i] + nKtrain[j] - 2 * Ktrain_test[i, j];

    indices = np.argsort(distMatrix, axis=0);
    preds = np.zeros(rows,"int64");

    for i in range(0, rows):
        counts = np.zeros(31);
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
    Ktrain_test = Ktrain_test + Ktrain_learn  @ s @ Ktest_learn.T
    nKtest = np.ones((Xtest.shape[0],1), 'float64');
    nKtrain = np.ones((Xtrain.shape[0],1), 'float64');

    # call knn
    predLabels = kernelKNN(Ytrain, Ktrain_test, nKtrain, nKtest, k);
    # compute accuracy
    numRight = np.sum(predLabels == Ytest)
    acc  = numRight / len(predLabels);
    print ("Matching completed with accuracy of "+ str(acc*100) +" %")
    return acc;


# if __name__=='__main__':
#     s = np.array(np.random.rand(684, 684))
#     k = 1
#     Xlearn = np.array(np.random.rand(684, 800))
#
#     trknnA = np.full(591, 1)
#     testexsB = np.full(498, 1)
#
#     xA = np.array(np.random.rand(795, 800))
#     xB = np.array(np.random.rand(498, 800))
#
#     yA = np.full(795, 1)
#     yB = np.full(498, 1)
#
#     print(asymmetricKNN(xA[trknnA, :], yA[trknnA], xB[testexsB, :], yB[testexsB], Xlearn, s, k))
#
#
