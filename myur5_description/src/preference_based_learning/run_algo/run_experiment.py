import numpy as np
from simulation_utils import get_feedback
import argparse
import matplotlib.pyplot as plt

from run_algo.algo_utils import define_algo
from run_algo.evaluation_metrics import cosine_metric



 
if __name__ == "__main__":

    
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--algo", type=str, default="DPB",
                    choices=['DPB', 'batch_active_PBL'], help="type of algorithm")
    ap.add_argument('-e', "--num-iteration", type=int, default=300,
                    help="# of iteration")
    ap.add_argument('-t', "--task-env", type=str, default="driver",
                    help="type of simulation environment")
    ap.add_argument('-b', "--num-batch", type=int, default=10,
                    help="# of batch")
    
    
    args = vars(ap.parse_args())
    
    b = args['num_batch']
    N = args['num_iteration']
    task = args['task_env']
    algo, true_w = define_algo(task, args['algo'])
    
    
    if N % b != 0:
        print('N must be divisible to b')
        exit(0)
    
    t = 0
    t_th_w = 0
    
    turning_point = 100
    eval_cosine = [0]
    
    
    
    
    while t < N:
        print('Samples so far: ' + str(t))
        
        if t!=0 and t%(turning_point)==0:
            t_th_w+=1
            
            
        algo.update_param(t)
        actions = algo.select_batch_actions(t, b)
        
        
        # simulated human
        for i in range(b):
            A, R = get_feedback(actions[i], true_w[t_th_w], m="samling")
            
            algo.actions_s.append(A)
            algo.reward_s.append(R)
            
            t+=1
            
        eval_cosine.append(cosine_metric(algo.hat_theta_D, true_w[t_th_w]))
        
    
    
    plt.plot(b*np.arange(len(eval_cosine)), eval_cosine, color='orange', label='base', alpha=0.8)
    plt.show()
        
    
    
    
    
    
    
    # for g in range(25):
        
    #     for a in range(30):6
            
            
    #         DPB_params['gamma'] = (g+1)*0.01 + 0.7
    #         DPB_params['exploration_weight'] = (a+1)*0.01
            
            
    #         bench_experiment("driver", "greedy",
    #                         N=300, b=10, 
    #                         batch_active_params=batch_active_params,
    #                         DPB_params=DPB_params,
    #                         num_randomseeds=5)