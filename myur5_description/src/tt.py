import numpy as np
import os
dir_path = os.path.dirname(os.path.abspath(__file__))
yy = os.path.join(dir_path,  'sampled_trajectories/planning_trajectory.npz')
planning_trajectory = np.load(yy, allow_pickle=True)


k = []

print(type(k))


# print(len(planning_trajectory['plan']))
# print(planning_trajectory['plan'][0])
