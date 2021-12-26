import numpy as np


def revolute_degree(x):
    y = np.array([1, 0, 0])
    z = np.zeros(3)
    z[0] = x[1]*y[2]-x[2]*y[1]
    z[1] = x[2]*y[0]-x[0]*y[2]
    z[2] = x[0]*y[1]-x[1]*y[0]
    


    return np.arcsin(np.linalg.norm(z)/(np.linalg.norm(x)*np.linalg.norm(y)))*np.sign(z[2])
a = [1, 2, 3, 4]

print(max(a))
print(revolute_degree(np.array([1, -1, 0])))
print(revolute_degree(np.array([1, 1, 0])))
