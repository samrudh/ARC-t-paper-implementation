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




def getConstraints_InterdomainSimilarity(y1,y2,l,u):
    pos = 1
    ly1 = len(y1)
    ly2 = len(y2)
    C = np.zeros(ly1 * ly2, 4)

    for i in range(ly1):
        for j in range(ly2):
            if y1[i] == y2[j]:
                C[pos,:] =[i, j + ly1, u ,-1]
            else:
                C[pos,:]=[i ,j + ly1, l, 1]

            pos = pos + 1

    m = ly1 + ly2
    indices = [x for x in range(m)]

    return indices


def learnAymmTransform(XA, yA, XB, yB, params):

    X = np.concatenate((XA, XB))
    y = np.concatenate((yA, yB))


    ## Form kernel matrix
    K0train = helper.formKernel(X, X)

    ## Calculate lowe and upper thresholds
    l, u = helper.getKernelValueExtremes(K0train, 0.02, 0.98)

    return getConstraints_InterdomainSimilarity(yA,yB,l,u)



















if __name__=='__main__':

    params = Params()

    xA = np.random.rand(795,800)
    xB = np.random.rand(498,800)

    yA = np.array([1]*795)
    yB = np.array([1] * 498)

    print(learnAymmTransform(xA, yA, xB, yB, params))


