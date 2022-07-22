from simulation_utils import create_env, perform_best
import sys
import numpy as np

def find_opt_trj(simulation_object, w):


    iter_count = 5 
    
    data = np.load('ctrl_samples/' + simulation_object.name + '.npz')
    actions = data['psi_set']

    print(perform_best(simulation_object, w, iter_count))
    #opt_trj = np.argmax(np.dot(actions, w))
    

    return 
