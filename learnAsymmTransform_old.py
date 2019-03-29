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
    
class Trasnform():





    def learnAymmTransform(XA, yA, XB, yB, params):
    #    debug
    #    XA = xA
    #    XB = xB
        X = np.concatenate((XA, XB))
        y = np.concatenate((yA, yB))
    
    
        ## Form kernel matrix
        K0train = helper.formKernel(X, X)
    
        ## Calculate lowe and upper thresholds
        l, u = helper.getKernelValueExtremes(K0train, 0.02, 0.98)
    
        C, indices = helper.getConstraints_InterdomainSimilarity(yA,yB,l,u)
    
        # S, slack  = helper.asymmetricFrob_slack_kernel(K0train, C)
    
        return helper.asymmetricFrob_slack_kernel(K0train, C)



















#if __name__=='__main__':
#
#    params = Params()
#
#    xA = np.random.rand(591,800)
#    xB = np.random.rand(93,800)
#
#    yA = np.array([1]*591)
#    yB = np.array([1] * 93)
#
#    print(learnAymmTransform(xA, yA, xB, yB, params))


