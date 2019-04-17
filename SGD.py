# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 14:52:06 2019

@author: SHREYANK
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def SGD(K0,C,gamma=100,thresh=0.01, iterations=100):    
     weights=np.random.randn(K0.shape[0],K0.shape[1])
     lr=thresh
     n, _ = K0.shape
     C[:,0] = np.array([int(x) for x in list(C[:,0])])
     C[:,1] = np.array([int(x) for x in list(C[:,1])])
    
     v = n*(C[:, 0] ) + C[:,1]
     v = np.array([int(x) for x in list(v)])
    
     k0_flatten = np.ravel(K0)
     k0_new = [k0_flatten[abs(int(x))] for x in v]
     k0_new = np.array(k0_new)
     viol = np.array(C[:,3] * ( k0_new - C[:,2] ))
     viol.reshape([1, len(viol)])
     for i in range(iterations):
         mx = max(viol)
         curri = np.argmax(viol)
         p1 = int(C[curri][0])
         p2 = int(C[curri][1])
         b = C[curri][2]
         s = C[curri][3]

         kx = K0[p1,:]
         ky = K0[:, p2]
         #LOSS
         t1 = (s * (b - K0[p1, p2] - np.matmul(np.matmul(kx.T ,S), ky)) )
         t2 = ( + K0[p1, p1] * K0[p2, p2])
         t = t1 / t2
         #cost,d_weight=derivative(C,weights)
         
         weights[p1, p2] = weights[p1, p2] - lr * t
         
         return weights
###


