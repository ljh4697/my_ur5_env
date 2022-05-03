import numpy as np

a = {}

a = np.arange(10, 20)
b = np.arange(30, 40)


c = np.concatenate((a[:5], b[:5]))
print(c)