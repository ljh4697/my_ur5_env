import numpy as np


estimate_w = [[0]for i in range(2)]

estimate_w[0].append(5)
estimate_w[1].append(10)


print(estimate_w)
print(np.mean(np.array(estimate_w), axis=0))