import os
import numpy as np


pick_trajectories = np.load('./sampled_trajectories/pick_trajectories.npz')

for i in range(1, 6):
    idx ="p" + f"{i}"
    print(pick_trajectories[idx])