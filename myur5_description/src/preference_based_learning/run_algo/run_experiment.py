#!/usr/bin/env python3

import numpy as np
from simulation_utils import get_feedback
import argparse
import matplotlib.pyplot as plt

from run_algo.algo_utils import define_algo
from run_algo.evaluation_metrics import cosine_metric, simple_regret, regret
from run_optimizer import get_opt_id


 
if __name__ == "__main__":

    
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--algo", type=str, default="DPB",
                    choices=['DPB', 'batch_active_PBL', 'DPB2'], help="type of algorithm")
    ap.add_argument('-e', "--num-iteration", type=int, default=400,
                    help="# of iteration")
    ap.add_argument('-t', "--task-env", type=str, default="driver",
                    help="type of simulation environment")
    ap.add_argument('-b', "--num-batch", type=int, default=10,
                    help="# of batch")
    ap.add_argument('-s' ,'--seed',  type=int, default=1, help='A random seed')
    ap.add_argument('-w' ,'--exploration-weight',  type=float, default=0.0002, help='DPB hyperparameter exploration weight')
    ap.add_argument('-g' ,'--discounting-factor',  type=float, default=0.952, help='DPB hyperparameter discounting factor')
    ap.add_argument('-d' ,'--delta',  type=float, default=0.7, help='DPB hyperparameter delta')
    ap.add_argument('-l' ,'--regularized-lambda',  type=float, default=0.1, help='DPB regularized lambda')
    ap.add_argument('-bm' ,'--BA-method',  type=str, default='greedy', help='method of batch active')
    
        

    
    args = vars(ap.parse_args())

    # random seed
    # seed 3 for tosser video
    seed = args['seed']
    
    np.random.seed(seed)
    
    
    algos_type = args['algo']
    b = args['num_batch']
    N = args['num_iteration']
    task = args['task_env']
    
    if N % b != 0:
        print('N must be divisible to b')
        exit(0)
    
    
    algo, true_w = define_algo(task, algos_type, args, 'simulated')
    feature_set = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/ctrl_samples/' + task + '_features'+'.npz', allow_pickle=True)['features']
    
    
    
    t = 0
    t_th_w = 0
    
    turning_point = 100
    eval_cosine = [0]
    opt_simple_reward = [0]
    eval_simple_regret = [0]
    eval_cumulative_regret = [0]
    
    #hat_theta_t = [algo.hat_theta_D] ; true_theta_t = [true_w[t_th_w]]
    
    
    while t < N:
        print('Samples so far: ' + str(t))
        
        
        # time varying true theta
        if t!= 0 and t%(turning_point)==0:
            if t!=300:
                t_th_w+=1
                
            #hat_theta_t.append(algo.hat_theta_D)
            #true_theta_t.append(true_w[t_th_w])
            
            
            
        # visualize opt traj
        #20, 70, 120, 150, 170, 220, 270, 350]
        
        # if int(t) in [20, 70, 120, 150, 170, 220, 270, 350]:
        #     #algo.hat_theta_D)
        #     print("round {}".format(str(t)))
        #     opt_id = get_opt_id(algo.predefined_features, algo.hat_theta_D)
        #     t_opt_id = get_opt_id(algo.predefined_features, true_w[t_th_w])
            
        #     print(np.dot(algo.predefined_features[opt_id], true_w[t_th_w]))
        #     print(np.dot(algo.predefined_features[t_opt_id], true_w[t_th_w]))
            
                        
        #     # total_num = algo.predefined_features.shape[0]
        #     # half_t_num = total_num/2
            
            
        #     # z = algo.simulation_object.feed_size
            
            
        #     # if opt_id<half_t_num:
        #     #     algo.simulation_object.feed(algo.inputs_set[opt_id,0:z])
        #     # else:
        #     #     opt_id = int(opt_id - half_t_num)
        #     #     algo.simulation_object.feed(algo.inputs_set[opt_id,z:2*z])
                
        #     # algo.simulation_object.watch(1)
                            
            
            
            
            
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
            
            A, R = get_feedback(algo, inputA_set[i], inputB_set[i],
                                actions[i], true_w[t_th_w], m="samling", human='simulated')
            
            if args['BA_method'] == 'information':
                
                if task == 'avoid':
                    #R = -R
                    idx = inputA_set[i]
                
                    phi_A = list(feature_set[idx*2])
                    phi_B = list(feature_set[idx*2+1])
                    # data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/ctrl_samples/' + algo.simulation_object.name + '.npz', allow_pickle=True)
                    # PSI = data['psi_set']
                    # print(phi_A-phi_B)
                    # print(PSI[idx])
                    # print(len(PSI))
                                    
                else:
                    algo.simulation_object.feed(inputA_set[i])
                    phi_A = algo.simulation_object.get_features()    
                    
                    algo.simulation_object.feed(inputB_set[i])
                    phi_B = algo.simulation_object.get_features()
                    
                    
                algo.feed(phi_A, phi_B, [-R])
                algo.action_s.append(A)
            else:
                algo.action_s.append(A)
                algo.reward_s.append(R)
                
            
            t+=1
            

            
    # print("round {}".format(str(t)))
    # opt_id = get_opt_id(algo.predefined_features, algo.hat_theta_D)
                
    # total_num = algo.predefined_features.shape[0]
    # half_t_num = total_num/2
    
    
    # z = algo.simulation_object.feed_size
    
    
    # if opt_id<half_t_num:
    #     algo.simulation_object.feed(algo.inputs_set[opt_id,0:z])
    # else:
    #     opt_id = int(opt_id - half_t_num)
    #     algo.simulation_object.feed(algo.inputs_set[opt_id,z:2*z])
        
    # algo.simulation_object.watch(5)
            
    
    
    
    #hat_theta_t.append(algo.hat_theta_D)
    #true_theta_t.append(true_w[t_th_w])
    
    
    
    
    
    # save result
    if algos_type == 'DPB':
        filename = '../results/{}/{}_greedy/{}-iter{:d}-{:}-delta{:.2f}-alpha{:.4f}-gamma{:.3f}-lambda{:.2f}-seed{:d}.npy'.format(task, algos_type, task, N, algos_type, args["delta"], args["exploration_weight"], args["discounting_factor"], args['regularized_lambda'], seed)
    elif algos_type == 'DPB2':
        filename = '../results/{}/{}/{}-iter{:d}-{:}-delta{:.2f}-alpha{:.4f}-gamma{:.3f}-lambda{:.2f}-seed{:d}.npy'.format(task, algos_type, task, N, algos_type, args["delta"], args["exploration_weight"], args["discounting_factor"], args['regularized_lambda'], seed)
    elif algos_type == "batch_active_PBL":
        filename = '../results/{}/{}/n{}-iter{:d}-{:}-method_{}-seed{:d}.npy'.format(task, algos_type, task, N, algos_type, args["BA_method"], seed)
    
    
    with open(filename, 'wb') as f:
        np.savez(f,
                eval_cosine=eval_cosine,
                eval_simple_regret=eval_simple_regret,
                opt_simple_reward=opt_simple_reward,
                eval_cumulative_regret=eval_cumulative_regret
                #hat_theta_t = hat_theta_t,
                #true_theta_t = true_theta_t
        )
        
        print('data saved at {}'.format(filename))
    
    
    