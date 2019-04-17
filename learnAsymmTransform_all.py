### Learning asymmetric transform
'''
Learning asymmetric transform

'''
import numpy as np
import math
import helper
import asymmetricKNN;
import SVM
from os import path
import symmetricKNN;
import SGD
class Params():
    def __init__(self):
        pass




import scipy.io as sio
dir_name='matlab_data/domain_adaptation_features_20110928'
#dir_name = path.join('matlab_data', 'domain_adaptation_features_20110928')
config_file1 = 'config_samecat_webcam_dslr'
config_file2 = 'config_samecat_dslr_webcam'
config_file3 = 'config_diffcat_webcam_dslr'
config_file4 = 'config_diffcat_dslr_webcam'
config_files = [config_file1, config_file2, config_file3, config_file4]

accuracies = {}
iterations =  1000

for config_file in config_files:

    mat_file_path = r"_matlab_input_data.mat"
    #file_name = path.join(dir_name , config_file + mat_file_path)
    file_name = dir_name+'/'+ config_file + mat_file_path
    matlab_inputs = sio.loadmat(file_name)
    
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
    
    S, slack  = helper.asymmetricFrob_slack_kernel(K0train, C, iterations)
    #S=SGD.SGD(K0train, C, iterations)
    
    #params.S = S
    Xlearn = X[indices,:]
    
    
    file_name = path.join(dir_name, config_file + "_knn_matlab_input_data.mat")
    matlab_inputs_knn = sio.loadmat(file_name)
    X1 = matlab_inputs_knn['out_XA']
    y1 = matlab_inputs_knn['out_yA']
    X2 = matlab_inputs_knn['out_XB']
    y2 = matlab_inputs_knn['out_yB']
    
    ## KNN code Umesh
   # accuracy = asymmetricKNN.asymmetricKNN(X1,y1,X2,y2,Xlearn,S,1);
    accuracy=SVM.SVM(X1,y1,X2,y2,Xlearn,S,1);
    accuracies[config_file] = accuracy
    
    

##
symmetricKNN.symmetricKNN(X1,y1,X2,y2,Xlearn,S,1);

## SVM code Samrudha
##asymmetricKNN.asymmetricSVM(X1,y1,X2,y2,Xlearn,S,1);



