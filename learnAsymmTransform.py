### Learning asymmetric transform
'''
Learning asymmetric transform

'''
import numpy as np
import math
import helper
import asymmetricKNN;

class Params():
    def __init__(self):
        pass




import scipy.io as sio

mat_file_path = r"matlab_input_data.mat"

matlab_inputs = sio.loadmat(mat_file_path)

XA = matlab_inputs['in_XA']
yA = matlab_inputs['in_yA']
XB = matlab_inputs['in_XB']
yB= matlab_inputs['in_yB']
param = matlab_inputs['in_param']





    
    
    
X = np.concatenate((XA, XB), axis=0)
y = np.concatenate((yA, yB), axis=1)


## Form kernel matrix
K0train = helper.formKernel(X, X)

## Calculate lowe and upper thresholds
l, u = helper.getKernelValueExtremes(K0train, 0.02, 0.98)

C, indices = helper.getConstraints_InterdomainSimilarity(yA,yB,l,u)

S, slack  = helper.asymmetricFrob_slack_kernel(K0train, C)


## start coding from here
### use
#params.S = S
Xlearn = X[indices,:]

matlab_inputs_knn = sio.loadmat("knn_matlab_input_data.mat")
X1 = matlab_inputs_knn['out_XA']
y1 = matlab_inputs_knn['out_yA']
X2 = matlab_inputs_knn['out_XB']
y2 = matlab_inputs_knn['out_yB']

## KNN code Umesh
asymmetricKNN.asymmetricKNN(X1,y1,X2,y2,Xlearn,S,1);













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


