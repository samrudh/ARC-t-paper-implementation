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




import scipy.io as sio

mat_file_path = r"matlab_input_data.mat"

matlab_inputs = sio.loadmat(mat_file_path)

XA = matlab_inputs['in_XA']
yA = matlab_inputs['in_yA']
XB = matlab_inputs['in_XB']
yB= matlab_inputs['in_yB']
param = matlab_inputs['in_param']


mat_file_path = r"knn_matlab_input_data.mat"

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
params.S = S
Xlearn = X[indices,:]
###
mat_file_path = r"knn_matlab_input_data.mat"

matlab_inputs = sio.loadmat(mat_file_path)

X1 = matlab_inputs['out_XA']
y1 = matlab_inputs['out_yA']
X2 = matlab_inputs['out_XB']
y2 = matlab_inputs['out_yB']


 asymmetricKNN(X1,y1,X2,y2,?);


















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


