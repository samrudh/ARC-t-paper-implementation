### Learning asymmetric transform
'''
Learning asymmetric transform

'''
import numpy as np
import math
import helper




class Params():


    def __init__(self):
        pass




def getConstraints_InterdomainSimilarity(yA,yB,l,u);


    pos = 1;
    ly1 = length(y1);
    ly2 = length(y2);
    C = zeros(ly1 * ly2, 4)

    pass


def learnAymmTransform(XA, yA, XB, yB, params):

    X = np.concatenate((XA, XB))
    y = np.concatenate((yA, yB))


    ## Form kernel matrix
    K0train = helper.formKernel(X, X)

    ## Calculate lowe and upper thresholds
    l, u = helper.getKernelValueExtremes(K0train, 0.02, 0.98)

    return True



















if __name__=='__main__':

    params = Params()

    xA = np.random.rand(795,800)
    xB = np.random.rand(498,800)

    yA = np.array([1]*795)
    yB = np.array([1] * 498)

    print(learnAymmTransform(xA, yA, xB, yB, params))


