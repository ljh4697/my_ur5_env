import matplotlib
import numpy as np
import matplotlib.pyplot as plt







a = np.array([])
a = np.append(a,  np.arange(3))

b = np.arange(10)

print(np.where(np.array(list(map(lambda x: x in a, b )))==1)[0])
print(1 in a)


