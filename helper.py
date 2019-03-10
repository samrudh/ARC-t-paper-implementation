import numpy as np
import math

def formKernel(X1, X2, gamma=None):
    return np.exp( np.subtract( np.matmul(X1,X2.T),1) )



def getKernelValueExtremes(K, lowerPct, upperPct):


    v1 = np.sort(K, axis =None)

    low_index = math.ceil(lowerPct * len(v1))
    high_index = math.ceil(upperPct * len(v1))

    l = v1[low_index];
    u = v1[high_index]

    return l, u
