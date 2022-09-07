import numpy as np


tr1 = np.load('./trajectory_ex/tosser/tj1.npz', allow_pickle=True)



print(tr1['human'].shape)
print(tr1['human'])
print(tr1['robot'].shape)
print(tr1['robot'])
