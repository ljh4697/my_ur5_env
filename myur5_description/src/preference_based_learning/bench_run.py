#from bench_demos_pre import bench_experiment
import numpy as np
from simulation_utils import create_env

##### hyper param ############
##############################
batch_active_params = {
    "samples_num":1000
}

##############################
DPB_params = {
    "exploration_weight":0.03,
    "discounting_factor":0.92,
    "action_U":1.4,
    "param_U":1,
    "regularized_lambda":0.1,
    "reward_U":1,
    "delta":0.7,
    "c_mu":1/5,
    "k_mu":1/4
}

############################


####### theta star ######
# theta_star = {
#     0:[np.random.uniform(0,0.1), np.random.uniform(0.9,0.99), 0.3, 0.2],
#     1:[np.random.uniform(0,0.1), np.random.uniform(0.9,0.99), 0.3, 0.2],
#     2:[-np.random.uniform(0.9,0.99), -np.random.uniform(0.9,0.99), 0.3, 0.2]
# } 






 
if __name__ == "__main__":
    
    b = 10
    d = 4
    simulation_object = create_env('driver')
    
    
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]
    
    inputA_set = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(b, 2*simulation_object.feed_size))
    inputB_set = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(b, 2*simulation_object.feed_size))
    
    psi = np.zeros((b, d))
    
    for i in range(b):
    
        simulation_object.feed(inputA_set[i])
        phi_A = simulation_object.get_features()    
        
        simulation_object.feed(inputB_set[i])
        phi_B = simulation_object.get_features()
        
        psi[i, :] = np.array(phi_A) - np.array(phi_B)
    
    print(psi)
    
    
    # for g in range(25):
        
    #     for a in range(30):6
            
            
    #         DPB_params['gamma'] = (g+1)*0.01 + 0.7
    #         DPB_params['exploration_weight'] = (a+1)*0.01
            
            
    #         bench_experiment("driver", "greedy",
    #                         N=300, b=10, 
    #                         batch_active_params=batch_active_params,
    #                         DPB_params=DPB_params,
    #                         num_randomseeds=5)