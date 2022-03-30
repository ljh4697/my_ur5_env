import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy


# true_w = np.random.rand(4)
# true_w = true_w/np.linalg.norm(true_w)
# print(true_w[0])


# def current_w(w, t):
#     n_w = copy.deepcopy(w)
#     n_w[0] += (1/t)*np.sin(t)
#     n_w = n_w/np.linalg.norm(n_w)
#     return n_w

# archives_w0 = [] 

# for t in range(1,101):
#     current_preference_w = current_w(true_w, t)
#     archives_w0.append(current_preference_w[0])
    

# print(true_w[0])
# plt.title('target_w_0')
# plt.plot(np.arange(1,101), archives_w0)
# plt.plot(np.arange(1,101), np.ones(100)*true_w[0], 'r--')
# plt.ylim(0,1)
# plt.xlabel('t')
# plt.ylabel('w0')
# plt.show()
estimate_w = [[0]for i in range(10)]
print(estimate_w)
for i in range(10):
    estimate_w[i].append(1)

print(estimate_w)
print(np.mean(np.array(estimate_w), axis=0))