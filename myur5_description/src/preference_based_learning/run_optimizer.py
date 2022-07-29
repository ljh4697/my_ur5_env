from simulation_utils import create_env, perform_best
import sys
import numpy as np

def get_opt_features(simulation_object, w):


    iter_count = 2
    

    best_score, best_trajectory = (perform_best(simulation_object, w, iter_count))
    
    simulation_object.set_ctrl(best_trajectory)
    opt_features = simulation_object.get_features()
    #opt_trj = np.argmax(np.dot(actions, w))

    return opt_features
