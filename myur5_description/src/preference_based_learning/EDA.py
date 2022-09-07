from simulation_utils import create_env, perform_best
import sys
import numpy as np


task = 'avoid'
simulation_object = create_env(task)
data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/ctrl_samples' +
                '/' + simulation_object.name + '.npz')
psi_set = data['psi_set']

print(task)

print(np.mean(psi_set, axis=0))
print(np.std(psi_set, axis=0))

