import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate fake data
x = np.random.normal(size=1000)
y = x * 3 + np.random.normal(size=1000)

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

fig, ax = plt.subplots()

true_w = np.array([0.36071584, -0.4335038 ])
target_w=np.array([0.21584, -0.7835038  ]) #ex1
plt.scatter(target_w[0],target_w[1], s=220, c='orange')
plt.scatter(true_w[0],true_w[1], s=220, c='blue')

plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel('w1')
plt.ylabel('w2')
plt.tight_layout()
plt.title('target w')
plt.show()