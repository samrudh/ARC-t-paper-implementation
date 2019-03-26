# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:31:59 2019

@author: SHREYANK
"""
import os
import cv2
from compute_sift_features import sift_features
import scipy.io as sio
import numpy as np

def read_data(domain_name):
    domain_dir=data_dir+histogram_surf_feats_dir+domain_name
    domain_feats=os.listdir(domain_dir)
    data=[]
    data_id=[]
    for i in range(len(domain_feats)):
        current_class=domain_feats[i]
        class_dir=domain_dir+current_class
        class_samples=os.listdir(class_dir)
        for j in range(len(class_samples)):  
            current_sample=class_samples[j]
            if current_sample.endswith('_800.SURF_SURF.mat'):
                sample_dir=class_dir+'/'+current_sample
                hist_feats=sio.loadmat(sample_dir)['histogram'][0]
                obj_id=sio.loadmat(sample_dir)['object_id'][0]
                data.append(hist_feats)
                data_id.append([int(obj_id),i])
    data=np.asarray(data)
    data_id=np.asarray(data_id)
    return data,data_id


data_dir='../dataset/'
images_dir='domain_adaptation_images/'
histogram_surf_feats_dir='domain_adaptation_features_20110928/'
decaf_feats_dir='domain_adaptation_decaf_features_20140430/'
sift_feats_dir='doman_adaptation_sift_features/'
domain_A='dslr/interest_points/'
domain_B='webcam/interest_points/'
domainA_data,domainA_id=read_data(domain_A)
domainB_data,domainB_id=read_data(domain_B)
matches=sio.loadmat(data_dir+'matches.mat')['Matches']

XA=domainB_data
yA=domainB_id[:,1]
XB=domainA_data
yB=domainA_id[:,1]
k=3
gamma_set=100

nA,dA=np.shape(XA)
nB,dB=np.shape(XB)
NUM_RUNS=1
classes=31
num_training_A=20
num_training_B=3


for j in range(NUM_RUNS):
    
    trexsA, testexsA, trexsB, testexsB=[],[],[],[]

    for i in range(classes):
        objA=np.where(yA==i)[0]
        tmp=domainB_id[objA,0]
        rindx=np.random.permutation(len(objA));
        trexsA=np.concatenate([trexsA,objA[rindx[0:min(len(objA),num_training_A)]]],axis=0)
        testexsA=np.concatenate([testexsA,objA],axis=0)
        
        objB=np.where(yB==i)[0]
        tmp=domainA_id[objB,0]
        rindx=np.random.permutation(len(objB));
        trexsB=np.concatenate([trexsB,objB[rindx[0:min(len(objB),num_training_B)]]],axis=0)
        testexsB=np.concatenate([testexsB,objB],axis=0)
        
        
    
    
    trknnA=trexsA
    MatchesTrain=[]#np.zeros([1,2])
    for kk in range(len(matches)):
        ii=matches[kk,0]
        jj=matches[kk,1]
        if len(np.where(trexsA==ii)[0])>0 and len(np.where(trexsB==jj)[0])>0:            
            #MatchesTrain[count_MatchesTrain,0]=np.where(trexsA==ii)[0][0]
            #MatchesTrain[count_MatchesTrain,1]=np.where(trexsB==jj)[0][0]
            MatchesTrain.append([np.where(trexsA==ii)[0][0],np.where(trexsB==jj)[0][0]])
    MatchesTrain=np.asarray(MatchesTrain)    
    constraint_num=np.square(len(trexsB))
    for g in range(len(gamma_set)):

        gamma=gamma_set(g)



