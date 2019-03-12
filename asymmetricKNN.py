### Perform KNN
'''
Performs KNN based classification on asymmetric test and training data

'''
import numpy as np
from helper import formKernel


class Params():
    s = ''
    k = 0
    Xlearn = ''

    def __init__(self, s, k, xlearn):
        self.s = s
        self.k = k
        self.Xlearn = xlearn


def kernelKNN (Ytrain, Ktrain_test, nKtrain, nKtest, k):
    # Incase labels starts with 0, Add one to each label to maintain consistency
    labelsUpdated = False;
    if min(Ytrain) == 0:
        Ytrain = Ytrain + 1;
        labelsUpdated = True;
    # compute distance
    (rows, cols) = Ktrain_test.shape
    distMatrix = np.zeros((rows, cols), 'float64');
    for i in range(1,rows):
        for j in range(1,cols):
            distMatrix[i,j] = nKtest[i] + nKtrain[j] - 2 * Ktrain_test[i, j];

    indices = np.argsort(distMatrix, axis=0);
    preds = np.zeros((rows,1),"int64");

    for i in range(0, rows):
        counts = np.zeros(31);
        for j in range(0, k):
            if Ytrain[indices[i,j]] > np.count_nonzero(counts):
                counts[Ytrain[indices[i,j]]] = 1;
            else:
                counts[Ytrain[indices[i,j]]] = counts[Ytrain[indices[i,j]]] + 1;

        preds[i] = np.argmax(counts);
    if labelsUpdated :
        preds = preds - 1;
    return preds

def asymmetricKNN(Xtrain, Ytrain, Xtest, Ytest, PARAM):
    # Xtrain -> datapoints in training domain (typically domain A)
    # Xtest -> datapoints in testing domain (typically domain B)
    # Ytrain -> labels in training domain corresponding to datapoints of training domain
    # Ytest -> labels in test domain corresponding to datapoints of test domain

    # Features combined together used while learning S
    Xlearn = PARAM.Xlearn;
    # optimised L matrix (represented as s)
    s = PARAM.s;
    if s is None:
        raise Exception('\'s\' is not defined');
    Ktrain_test  = formKernel(Xtrain, Xtest, PARAM)
    Ktest_learn  = formKernel(Xtest,  Xlearn, PARAM)
    Ktrain_learn = formKernel(Xtrain, Xlearn, PARAM)
    # @ is used for matrix multiplication
    Ktrain_test = Ktrain_test + Ktrain_learn  @ s @ Ktest_learn.T
    nKtest = np.ones((Xtest.shape[0],1), 'float64');
    nKtrain = np.ones((Xtrain.shape[0],1), 'float64');

    # call knn
    predLabels = kernelKNN(Ytrain, Ktrain_test.T, nKtrain, nKtest, k).T;
    # compute accuracy
    numRight = len(np.nonzero(np.equal(predLabels,Ytest)));
    acc  = numRight / len(predLabels);
    return acc;


if __name__=='__main__':
    s = np.array(np.random.rand(684, 684))
    k = 1
    Xlearn = np.array(np.random.rand(684, 800))
    params = Params(s, k, Xlearn)

    trknnA = np.array([1]*591)
    testexsB = np.array([1]*498)

    xA = np.array(np.random.rand(795, 800))
    xB = np.array(np.random.rand(498, 800))

    yA = np.array([1]*795)
    yB = np.array([1] * 498)

    print(asymmetricKNN(xA[trknnA, :], yA[trknnA], xB[testexsB, :], yB[testexsB], params))


