#!/usr/bin/env python3

import numpy as np
from simulation_utils import get_feedback
import argparse
import matplotlib.pyplot as plt

from run_algo.algo_utils import define_algo
from run_algo.evaluation_metrics import cosine_metric, simple_regret, regret
from run_optimizer import get_opt_id
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.decomposition import PCA
 
def kernel_se(_X1,_X2,_hyp={'gain':1,'len':1,'noise':1e-8}):
    hyp_gain = float(_hyp['gain'])**2
    hyp_len  = 1/float(_hyp['len'])
    pairwise_dists = cdist(_X2,_X2,'euclidean')
    K = hyp_gain*np.exp(-pairwise_dists ** 2 / (hyp_len**2))
    return K
 
 
 
if __name__ == "__main__":

    
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--algo", type=str, default="DPB",
                    choices=['DPB', 'batch_active_PBL', 'DPB2'], help="type of algorithm")
    ap.add_argument('-e', "--num-iteration", type=int, default=400,
                    help="# of iteration")
    ap.add_argument('-t', "--task-env", type=str, default="avoid",
                    help="type of simulation environment")
    ap.add_argument('-b', "--num-batch", type=int, default=10,
                    help="# of batch")
    ap.add_argument('-s' ,'--seed',  type=int, default=1, help='A random seed')
    ap.add_argument('-w' ,'--exploration-weight',  type=float, default=0.0002, help='DPB hyperparameter exploration weight')
    ap.add_argument('-g' ,'--discounting-factor',  type=float, default=0.94, help='DPB hyperparameter discounting factor')
    ap.add_argument('-d' ,'--delta',  type=float, default=0.7, help='DPB hyperparameter delta')
    ap.add_argument('-l' ,'--regularized-lambda',  type=float, default=0.42, help='DPB regularized lambda')
    ap.add_argument('-bm' ,'--BA-method',  type=str, default='greedy', help='method of batch active')
    
        



    args = vars(ap.parse_args())

    # random seed
    seed = args['seed']
    np.random.seed(seed)
    
    
    algos_type = args['algo']
    b = args['num_batch']
    N = args['num_iteration']
    task = args['task_env']
    
    if N % b != 0:
        print('N must be divisible to b')
        exit(0)
        
    
    algos_type = 'DPB'
    args['algo'] = 'DPB'
    DPB_algo, true_w = define_algo(task, algos_type, args, 'simulated')
    
    
    algos_type = 'DPB2'
    args['algo'] = 'DPB2'
    
    DPB2_algo, true_w = define_algo(task, algos_type, args, 'simulated')
    

    t = 0
    t_th_w = 0

    det_mean_DPB2 = 0
    det_mean_DPB = 0
    while t < N:

        print('Samples so far: ' + str(t))
        
        # # time varying true theta
        # if t!=0 and t%(turning_point)==0:
        #     t_th_w+=1
            
            

        
        # # evaluation 
        # if t != 0:
        #     eval_cosine.append(cosine_metric(algo.hat_theta_D, true_w[t_th_w]))
        #     s_r, opt_reward = simple_regret(algo.predefined_features, algo.hat_theta_D, true_w[t_th_w])
            
        #     opt_simple_reward.append(opt_reward)
        #     eval_simple_regret.append(s_r)
        #     eval_cumulative_regret.append(eval_cumulative_regret[-1] + regret(algo.PSI , np.array(algo.action_s[-1:-b-1:-1]), true_w[t_th_w]))

        DPB2_algo.update_param(t)
        DPB2_actions, DPB2_inputA_set, DPB2_inputB_set = DPB2_algo.select_batch_actions(t, b)
        

        
        print('------------------')
        DPB_algo.update_param(t)
        DPB_actions, DPB_inputA_set, DPB_inputB_set = DPB_algo.select_batch_actions(t, b)
        
        #  human feedback
        

        DPB2_points = []
        DPB_points = []
        
        
        # DPB2
        for i in range(b):
            
            print('DPB2 algorithm query')
            DPB2_A, DPB2_R = get_feedback(DPB2_algo, DPB2_inputA_set[i], DPB2_inputB_set[i],
                                DPB2_actions[i], true_w[t_th_w], m="samling", human='simulated')
            
            DPB2_points.append((DPB2_A[0], DPB2_A[1]))
            
            
            DPB2_algo.action_s.append(DPB2_A)
            DPB2_algo.reward_s.append(DPB2_R)
            

            
        # DPB
        for i in range(b):
            
            
            print('DPB algorithm query')
            DPB_A, DPB_R = get_feedback(DPB_algo, DPB_inputA_set[i], DPB_inputB_set[i],
                                DPB_actions[i], true_w[t_th_w], m="samling", human='simulated')
            
            
            DPB_points.append((DPB_A[0], DPB_A[1]))
            
            DPB_algo.action_s.append(DPB_A)
            DPB_algo.reward_s.append(DPB_R)
            DPB_algo.compute_V_t(DPB_A)
            
            t+=1
            
        DPB2_points = np.array(DPB2_points)
        DPB_points = np.array(DPB_points)
        
        
        

        
        
        print('----------------------')
        DPB2_algo_action_s = np.array(DPB2_algo.action_s)[t-b:t]
        mid_dist = np.median(cdist(DPB2_algo_action_s,DPB2_algo_action_s,'euclidean'))
        DPB2_K = kernel_se(DPB2_algo_action_s,DPB2_algo_action_s,{'gain':1,'len':mid_dist,'noise':1e-4})
        det_mean_DPB2 += np.linalg.det(DPB2_K)
        
        DPB_algo_action_s = np.array(DPB_algo.action_s)[t-b:t]
        mid_dist = np.median(cdist(DPB_algo_action_s,DPB_algo_action_s,'euclidean'))
        DPB_K = kernel_se(DPB_algo_action_s,DPB_algo_action_s,{'gain':1,'len':mid_dist,'noise':1e-4})
        det_mean_DPB += np.linalg.det(DPB_K)
        
        print('----------------------')
        
        pca = PCA(n_components=2, random_state=1004)
        DPB2_algo_action_s_pca = pca.fit_transform(DPB2_algo_action_s)
        DPB_algo_action_s_pca = pca.fit_transform(DPB_algo_action_s)
        
        
        x = list(DPB_algo_action_s_pca[:, 0]) + list(DPB2_algo_action_s_pca[:, 0])
        y = list(DPB_algo_action_s_pca[:, 1]) + list(DPB2_algo_action_s_pca[:, 1])
        
        
        plt.plot(DPB_algo_action_s_pca[:, 0], DPB_algo_action_s_pca[:, 1], 'mD',  alpha=0.8, label='DPB')
        plt.tight_layout()    
        
        plt.xlabel('PCA 1', fontsize= 15)
        plt.ylabel('PCA 2', fontsize= 15)
        plt.xticks(np.arange(min(x)-0.5, max(x)+0.5,2), fontsize= 15)
        plt.yticks(np.arange(min(y)-0.5, max(y)+0.5), fontsize= 15)
        plt.legend(fontsize=15)
        if t == 160:
            plt.savefig('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/results/imgs/avoid_psi_point_DPB.png', bbox_inches = 'tight')
        plt.show()
        

        plt.plot(DPB2_algo_action_s_pca[:, 0], DPB2_algo_action_s_pca[:, 1], 'bo', alpha=0.8, label='DPB (adaptive)')
        plt.tight_layout()    
        
        plt.xlabel('PCA 1', fontsize= 15)
        plt.ylabel('PCA 2', fontsize= 15)
        plt.xticks(np.arange(min(x)-0.5, max(x)+0.5,2), fontsize= 15)
        plt.yticks(np.arange(min(y)-0.5, max(y)+0.5), fontsize= 15)
        plt.legend(fontsize=15)
        
        
        if t == 160:
            plt.savefig('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/results/imgs/avoid_psi_point_DPB_adaptive.png', bbox_inches = 'tight')
        plt.show()
        
    print('----------------------')
    print(det_mean_DPB2)
    print(det_mean_DPB)




    # # save result
    # if algos_type == 'DPB':
    #     filename = '/home/joonhyeok/catkin_ws/src/doosan-robot/doosan-robot/dsr_example/py/scripts/PBL/preference_based_learning/results/{}/{}/{}-iter{:d}-{:}-delta{:.2f}-alpha{:.4f}-gamma{:.2f}-lambda{:.4f}-seed{:d}.npy'.format(task, algos_type, task, N, algos_type, args["delta"], args["exploration_weight"], args["discounting_factor"], args["regularized_lambda"], seed)
    #     labelname = '/home/joonhyeok/catkin_ws/src/doosan-robot/doosan-robot/dsr_example/py/scripts/PBL/preference_based_learning/results/{}/query_label/{}-iter{:d}-{:}-delta{:.2f}-alpha{:.4f}-gamma{:.2f}-lambda{:.4f}-seed{:d}.npy'.format(task, task, N, algos_type, args["delta"], args["exploration_weight"], args["discounting_factor"], args["regularized_lambda"], seed)
    # elif algos_type == "batch_active_PBL":
    #     filename = '/home/joonhyeok/catkin_ws/src/doosan-robot/doosan-robot/dsr_example/py/scripts/PBL/preference_based_learning/results/avoid/batch_active_PBL/{}-iter{:d}-{:}-method_{}-seed{:d}.npy'.format(task, N, algos_type, args["BA_method"], seed)
    #     labelname = '/home/joonhyeok/catkin_ws/src/doosan-robot/doosan-robot/dsr_example/py/scripts/PBL/preference_based_learning/results/avoid/query_label/{}-iter{:d}-{:}-method_{}-seed{:d}.npy'.format(task, N, algos_type, args["BA_method"], seed)
    

