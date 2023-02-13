import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist
print ("Packages loaded.")



def kernel_se(_X1,_X2,_hyp={'gain':1,'len':1,'noise':1e-8}):
    hyp_gain = float(_hyp['gain'])**2
    hyp_len  = 1/float(_hyp['len'])
    pairwise_dists = cdist(_X2,_X2,'euclidean')
    K = hyp_gain*np.exp(-pairwise_dists ** 2 / (hyp_len**2))
    return K
def kdpp(_X,_k):
    # Select _n samples out of _X using K-DPP
    n,d = _X.shape[0],_X.shape[1]
    mid_dist = np.median(cdist(_X,_X,'euclidean'))
    out,idx = np.zeros(shape=(_k,d)),[]
    for i in range(_k):
        if i == 0:
            rand_idx = np.random.randint(n)
            idx.append(rand_idx) # append index
            out[i,:] = _X[rand_idx,:] # append  inputs
        else:
            det_vals = np.zeros(n)
            for j in range(n):
                if j in idx:
                    det_vals[j] = -np.inf
                else:
                    idx_temp = idx.copy()
                    idx_temp.append(j)
                    X_curr = _X[idx_temp,:]
                    K = kernel_se(X_curr,X_curr,{'gain':1,'len':mid_dist,'noise':1e-4})
                    det_vals[j] = np.linalg.det(K)
            max_idx = np.argmax(det_vals)
            idx.append(max_idx)
            out[i,:] = _X[max_idx,:] # append  inputs
    return out,idx

# Data
n = 200
k = 10
x = np.random.rand(n,2)
# K-DPP and randomm sample 
kdpp_out,_ = kdpp(_X=x,_k=k)
rand_out = x[np.random.permutation(n)[:k],:]
# Plot
plt.figure(figsize=(6,6))
plt.plot(x[:,0],x[:,1],'k.')
hr, = plt.plot(rand_out[:,0],rand_out[:,1],marker='o',mec='blue',mfc='None',markersize=15,linestyle='None',mew=4)
hk, = plt.plot(kdpp_out[:,0],kdpp_out[:,1],marker='o',mec='red',mfc='None',markersize=15,linestyle='None',mew=4)
plt.xlim(0.0,1.0); plt.ylim(0.0,1.0)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend([hr,hk],['Random sample','K-DPP'],fontsize=15,loc='upper right')
plt.show()