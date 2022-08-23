
from simulation_utils import create_env, perform_best
import sys
import numpy as np
from tqdm import trange

def generate_features(simulation_object, inputs_set):
    z = simulation_object.feed_size
    inputs_set = np.array(inputs_set)
    if len(inputs_set.shape) == 1:
        inputs1 = inputs_set[0:z].reshape(1,z)
        inputs2 = inputs_set[z:2*z].reshape(1,z)
        input_count = 1
    else:
        inputs1 = inputs_set[:,0:z]
        inputs2 = inputs_set[:,z:2*z]
        input_count = inputs_set.shape[0]
    d = simulation_object.num_of_features
    features1 = np.zeros([input_count, d])
    features2 = np.zeros([input_count, d])  
    for i in trange(input_count):
        simulation_object.feed(list(inputs1[i]))
        features1[i] = simulation_object.get_features()
        simulation_object.feed(list(inputs2[i]))
        features2[i] = simulation_object.get_features()
        
        
    total_features = np.concatenate((features1, features2), axis=0)
    np.savez('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/ctrl_samples/' + simulation_object.name + '_features'+'.npz', 
             features = total_features
            )
    
    return total_features



if __name__ == "__main__":
    task = 'lunarlander'
    simulation_object = create_env(task)
    data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/ctrl_samples' +
                   '/' + simulation_object.name + '.npz')
    inputs_set = data['inputs_set']
    
    generate_features(simulation_object, inputs_set)
    