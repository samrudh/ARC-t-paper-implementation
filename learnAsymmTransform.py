### Learning asymmetric transform
'''
Learning asymmetric transform

'''
import numpy as np
import pandas as pd



class Params():


    def __init__(self):
        pass

def formKernel(X1, X2, gamma=None):
    return np.exp( np.subtract( np.matmul(X1,X2.T),1) )



def getKernelValueExtremes(K, lowerPct, upperPct):

    return np.sort(K, axis =None)



def learnAymmTransform(XA, yA, XB, yB, params):

    X = np.concatenate((XA, XB))
    y = np.concatenate((yA, yB))


    ## Form kernel matrix
    K0train = formKernel(X, X)

    ## Calculate lowe and upper thresholds
    a = getKernelValueExtremes(K0train, 0.02, 0.98)

    return a



















if __name__=='__main__':

    params = Params()

    xA = np.array(np.random([795,800]))
    xB = np.array(np.random([498, 800]))

    yA = np.array([1]*795)
    yB = np.array([1] * 498)

    print(learnAymmTransform(xA, yA, xB, yB, params))


