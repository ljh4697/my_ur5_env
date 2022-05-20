import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.metrics.pairwise import pairwise_distances
import demos








a = np.array([0.1, 0.5])
b = np.array([1, 1])

d = [1, 1]


#print(demos.get_entropy(b, a))
c = np.arange(5,0, -1)
print(c)
# print(np.argsort(c))
#print(np.delete(c,np.argsort(c)[:2]))


b = [5,4,3,2,1]
#for i in np.argsort(c)[:2]:
#    del b[i]


a = np.array([[1,1], [2,2], [3, 3]])
b = np.array([0, 1, 2])
print(a)
print(np.delete(a , np.argsort(-b)[0], axis=0))
