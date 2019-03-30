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

matlab_inputs_asymmetric = sio.loadmat(r"input_asymmetri_transform.mat")
testexsA = matlab_inputs_asymmetric['testexsA']
testexsB = matlab_inputs_asymmetric['testexsB']
trexsA = matlab_inputs_asymmetric['trexsA']
trexsB = matlab_inputs_asymmetric['trexsB']
trknnA = matlab_inputs_asymmetric['trknnA']

# Adjusting index by subtracting 1
testexsB = [[y - 1 for y in x] for x in testexsB];
trknnA = [[y - 1 for y in x] for x in trknnA];

X1 = matlab_inputs_asymmetric['XA']
y1 = matlab_inputs_asymmetric['yA'].reshape(-1)
X2 = matlab_inputs_asymmetric['XB']
y2 = matlab_inputs_asymmetric['yB'].reshape(-1)

## KNN code Umesh
asymmetricKNN.asymmetricKNN(X1[trknnA],y1[trknnA],X2[testexsB],y2[testexsB],Xlearn,S,1);

# Xtrain = X1[trknnA]
# Ytrain= y1[trknnA]
# Xtest = X2[testexsB];
# Ytest = y2[testexsB];
# s =S
# k =1
#













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


