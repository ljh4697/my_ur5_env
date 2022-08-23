import os
import numpy as np


da = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/ctrl_samples/driver.npz')

print(len(da['psi_set']))
