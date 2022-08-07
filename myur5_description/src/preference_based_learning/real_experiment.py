from isort import file
import numpy as np
from simulation_utils import get_feedback
import argparse
import matplotlib.pyplot as plt

from run_algo.algo_utils import define_algo
from run_algo.evaluation_metrics import cosine_metric, simple_regret, regret


 
if __name__ == "__main__":

    
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--algo", type=str, default="batch_active_PBL",
                    choices=['DPB', 'batch_active_PBL'], help="type of algorithm")
    ap.add_argument('-e', "--num-iteration", type=int, default=300,
                    help="# of iteration")
    ap.add_argument('-t', "--task-env", type=str, default="driver",
                    help="type of simulation environment")
    ap.add_argument('-b', "--num-batch", type=int, default=10,
                    help="# of batch")
    ap.add_argument('-s' ,'--seed',  type=int, default=1, help='A random seed')

    args = vars(ap.parse_args())


    #random seed
    seed = args['seed']
    np.random.seed(seed)
    
    
    algos_type = args['algo']
    b = args['num_batch']
    N = args['num_iteration']
    task = args['task_env']
    
    print(task)
    
    algo, true_w = define_algo(task, algos_type)
    

    if N % b != 0:
        print('N must be divisible to b')
        exit(0)
    
    t = 0
    t_th_w = 0
    
    turning_point = 100
    eval_cosine = [0]
    opt_simple_reward = [0]
    eval_simple_regret = [0]
    eval_cumulative_regret = [0]
    
    
    
    while t < N:
        print('Samples so far: ' + str(t))
        #print(eval_cosine)
        
        
        if t!=0 and t%(turning_point)==0:
            t_th_w+=1
            
            
        algo.update_param(t)
        actions, inputA_set, inputB_set = algo.select_batch_actions(t, b)
        
        # evaluation 
        if t != 0:
            eval_cosine.append(cosine_metric(algo.hat_theta_D, true_w[t_th_w]))
            s_r, opt_reward = simple_regret(algo.predefined_features, algo.hat_theta_D, true_w[t_th_w])
            
            opt_simple_reward.append(opt_reward)
            eval_simple_regret.append(s_r)
            eval_cumulative_regret.append(eval_cumulative_regret[-1] + regret(algo.PSI , np.array(algo.action_s[-1:-b-1:-1]), true_w[t_th_w]))
            
            
        #  human feedback
        for i in range(b):
            
            A, R = get_feedback(algo.simulation_object, inputA_set[i], inputB_set[i],
                                actions[i], true_w[t_th_w], m="samling", human='real')
            
            algo.action_s.append(A)
            algo.reward_s.append(R)
            
            t+=1
    
    
    filename = '../results/iter{:d}-{:}-seed{:d}.npy'.format(N, algos_type, seed)
    
    
    with open(filename, 'wb') as f:
        np.savez(f,
                eval_cosine=eval_cosine,
                eval_simple_regret=eval_simple_regret,
                opt_simple_reward=opt_simple_reward,
                eval_cumulative_regret=eval_cumulative_regret)
        
        print('data saved at {}'.format(filename))
    
    
    