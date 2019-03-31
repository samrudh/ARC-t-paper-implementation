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

mat_file_path = r"matlab_input_combined.mat"

matlab_inputs = sio.loadmat(mat_file_path)

XA = matlab_inputs['XA']
yA = matlab_inputs['yA'].reshape(-1)
XB = matlab_inputs['XB']
yB= matlab_inputs['yB'].reshape(-1)
param = matlab_inputs['PARAM']
trexsA = matlab_inputs['trexsA'].reshape(-1)
trexsB = matlab_inputs['trexsB'].reshape(-1)

trknnA = matlab_inputs['trknnA'].reshape(-1)
testexsB = matlab_inputs['testexsB'].reshape(-1)

# Adjusting index by subtracting 1
trexsA = [x - 1 for x in trexsA]
trexsB = [x - 1 for x in trexsB]
trknnA = [x-1 for x in trknnA]
testexsB = [x-1 for x in testexsB]

XA_learn = XA[trexsA,:]
yA_learn = yA[trexsA]
XB_learn = XB[trexsB,:]
yB_learn = yB[trexsB]


X = np.concatenate((XA_learn, XB_learn), axis=0)
y = np.concatenate((yA_learn, yB_learn), axis=0)


## Form kernel matrix
K0train = helper.formKernel(X, X)

## Calculate lowe and upper thresholds
l, u = helper.getKernelValueExtremes(K0train, 0.02, 0.98)

C, indices = helper.getConstraints_InterdomainSimilarity(yA_learn,yB_learn,l,u)

S, slack  = helper.asymmetricFrob_slack_kernel(K0train, C)


## start coding from here
### use
#params.S = S
Xlearn = X[indices,:]

## KNN code Umesh
asymmetricKNN.asymmetricKNN(XA[trknnA,:],yA[trknnA],XB[testexsB,:],yB[testexsB],Xlearn,S,1);

# For Debug purpose  only
# Xtrain = XA[trknnA,:]
# Ytrain= yA[trknnA]
# Xtest = XB[testexsB,:]
# Ytest = yB[testexsB]
# s =S
# k =1














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


