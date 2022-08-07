import numpy as np
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
test = os.path.join(dir_path, 'ctrl_samples/tosser.npz')

sampled = np.load(test, allow_pickle=True)

psi_set = np.zeros(shape=(0,4)) 
psi_set = np.array([[1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [3, 3, 3, 3]])

w = np.array([[1,1,1,1], [2,2,2,2]])


print(w.T)
#print(psi_set.dot(w))

print(psi_set.dot(w.T))
term1 = np.sum(1.-np.exp(-np.maximum(psi_set.dot(w.T),0)),axis=1)
print(-np.maximum(psi_set.dot(w.T),0))

