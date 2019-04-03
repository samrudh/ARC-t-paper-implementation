import numpy as np
from helper import formKernel
from asymmetricKNN import kernelKNN


def symmetricKNN(Xtrain, Ytrain, Xtest, Ytest, k):

    # a kernel matrix whose ij entry is the kernel fn between test point i and training point j
    Ktrain_test= formKernel(Xtrain, Xtest)

    nKtrain = np.ones(Xtrain.shape[0])
    nKtest = np.ones(Xtest.shape[0])
    # the self similarity of the training points (K_ii)
    for i in range(0, Xtrain.shape[0]):
        nKtrain[i] = formKernel(Xtrain[i, :],Xtrain[i, :])

    # the self similarity of the test points
    for i in range(0, Xtest.shape[0]):
        nKtest[i] = formKernel(Xtest[i, :], Xtest[i, :])

    # call knn
    predLabels = kernelKNN(Ytrain, Ktrain_test.T, nKtrain, nKtest, k);
    # compute accuracy
    numRight = np.sum(predLabels == Ytest)
    acc = numRight / len(predLabels);
    print ("Symmetric Matching completed with accuracy of " + str(acc*100) +" %")
    return acc;
