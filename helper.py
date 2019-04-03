import numpy as np
import math

def formKernel(X1, X2, gamma=None):
    return np.exp( np.subtract( np.matmul(X1,X2.T),1) )



def getKernelValueExtremes(K, lowerPct, upperPct):


    v1 = np.sort(K, axis =None)

    low_index = math.ceil(lowerPct * len(v1))
    high_index = math.ceil(upperPct * len(v1))

    l = v1[low_index]
    u = v1[high_index]

    return l, u



def getConstraints_InterdomainSimilarity(y1,y2,l,u):
    
    ##debug
    #y1,y2,l,u =  yA,yB,l,u
    ##debug
    
    pos = 0
    ly1 = y1.shape[1]
    ly2 = y2.shape[1]
    C = np.empty([(ly1*ly2), 4])

    for i in range(ly1):
        for j in range(ly2):
            if y1[0,i] == y2[0,j]:
                C[pos,:] =[i, j + ly1, u ,-1]
            else:
                C[pos,:]=[i ,j + ly1, l, 1]

            pos = pos + 1

    m = ly1 + ly2
    indices = [x for x in range(m)]

    return C, indices


def  asymmetricFrob_slack_kernel(K0,C,gamma=100,thresh=0.01, iterations=5000):
    
#    debug
#    K0,C, gamma, thresh=K0train, C , 100, 0.01
##   debug
    
    maxit = iterations
    n, _ = K0.shape
    S = np.zeros(K0.shape)
    c, t = C.shape
    slack = np.zeros([c, 1])
    lambda1 = np.zeros([c, 1])
    lambda2 = np.zeros([c, 1])
    
    C[:,0] = np.array([int(x) for x in list(C[:,0])])
    C[:,1] = np.array([int(x) for x in list(C[:,1])])
    
    v = n*(C[:, 0] ) + C[:,1]
    v = np.array([int(x) for x in list(v)])
    
    k0_flatten = np.ravel(K0)
    k0_new = [k0_flatten[abs(int(x))] for x in v]
    k0_new = np.array(k0_new)
    #
    viol = np.array(C[:,3] * ( k0_new - C[:,2] ))
    viol.reshape([1, len(viol)])

    for i in range(maxit):
        
        mx = max(viol)
        curri = np.argmax(viol)
        if not i%1000:
            print('Iteration ' , i,' maxviol ', mx)
            if mx < thresh:
                break

        p1 = int(C[curri][0])
        p2 = int(C[curri][1])
        b = C[curri][2]
        s = C[curri][3]

        kx = K0[p1,:]
        ky = K0[:, p2]

#        zz = 
        t1 = (s * (b - K0[p1, p2] - np.matmul(np.matmul(kx.T ,S), ky)) - slack[curri])
        t2 = (1 / gamma + K0[p1, p1] * K0[p2, p2])
        t = t1 / t2
        alpha = min(lambda1[curri], t)
        lambda1[curri] = lambda1[curri] - alpha
        S[p1, p2] = S[p1, p2] + s * alpha
        slack[curri] = slack[curri] - alpha / gamma
        alpha2 = min(lambda2[curri], gamma * slack[curri])
        slack[curri] = slack[curri] - alpha2 / gamma
        lambda2[curri] = lambda2[curri] - alpha2
        
        zz =C[:, 3].T
        # # updat viols
        v = np.zeros([len(C), 1])
        for x in range(C.shape[0]):
             v [x,0]= K0[int(C[x, 0]), p1]
             
        w = np.zeros([1,len(C)])        
        for x in range(C.shape[0]):
             w [0,x]= K0[p2,int(C[x, 1])]
        w=w.T
             
        zz = s*alpha*C[:,3]
        zz = np.reshape(zz,[len(zz),1])
        zz = zz.T
        zz1 = np.multiply(zz,v.T)
        zzz = np.multiply(zz1,w.T)
  
        viol = viol + zzz[0];
        viol[curri-1] = viol[curri] + (alpha+alpha2)/gamma

    return S, slack


