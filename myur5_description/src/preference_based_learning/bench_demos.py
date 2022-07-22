from sampling import Sampler
import bench_algos
import numpy as np
from simulation_utils import create_env, get_feedback, run_algo, bench_run_algo, bench_get_feedback
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from bandit_base import GLUCB
from scipy.optimize import fmin_slsqp
from run_optimizer import find_opt_trj

#true_w = [0.29754784,0.03725074,0.00664673,0.80602143]




def mu(x, theta):
    return 1/(1+np.exp(-np.dot(x, theta)))


def change_w_element(true_w):
    
    
    n_w = copy.deepcopy(true_w)
    
    max_id = np.argmax(n_w)
    min_id = np.argmin(n_w)
    
    max_v = n_w[max_id]
    min_v = n_w[min_id]
    
    n_w[max_id] = min_v
    n_w[min_id] = max_v
    
    
    return n_w

def batch(task, method, N, M, b):
    gamma = 0.92
    S = 1
    L = 1.4
    m = 1
    regularized_lambda = 0.1
    
    
    turning_point = int(N/3)
    tp = 0
    def regularized_log_likelihood(theta):
    
        return -(np.sum(np.array(gamma**np.arange(t,0,-1))*(np.array(reward_s)*np.log(mu(actions_s, theta))
                                                    +(1-np.array(reward_s))*np.log(1-mu(actions_s, theta))))-(regularized_lambda/2)*np.linalg.norm(theta)**2)
    

    def ieq_const(theta):
        return S-np.linalg.norm(theta)


    def get_true_reward(action, param_star):
        return np.dot(action,param_star)
    
    e = 1
    B = 20*b
    if N % b != 0:
        print('N must be divisible to b')
        exit(0)
    
    estimate_w_o = [[0]for i in range(e)]
    cosine_estimate_w = [[0]for i in range(e)]
    cosine_estimate_w_d = [[0]for i in range(e)] #
    
    true_reward_w = [[0]for i in range(e)]
    true_reward_w_d = [[0]for i in range(e)] #
    true_reward = [[0]for i in range(e)]
    
    simulation_object = create_env(task)
    
    # get psi set
    data = np.load('ctrl_samples/' + simulation_object.name + '.npz')
    actions = data['psi_set']
    #trajectory_set = data['inputs_set']
    
    
    
    
    

    features_d = simulation_object.num_of_features
    d = features_d
    for ite in range(e):

        
        # evaluate archive
        s_set = []
        reward_s = []
        actions_s = []
        
        
        psi_set = []
        s_set = []
        
        
        hat_theta_t = []
        base_theta_t = []
        true_theta_t = []
        # #

        # initialize
        

        
        # true_w = np.zeros(4)
        # true_w[2] = 0.15 ; true_w[3] = 0.2
        true_w = list(np.random.rand(features_d))
        
        true_w[0] = np.random.uniform(0,0.1)
        true_w[1] = np.random.uniform(0.9,0.99)
        true_w = true_w/np.linalg.norm(true_w)
        target_w = change_w_element(true_w)
        
        #print(simulation_object.ctrl_size)
        #find_opt_trj(simulation_object, target_w)
        
        D_PBL= GLUCB(d=features_d)
        
        
        
        lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
        upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

        w_sampler = Sampler(d)
        
        
 
        inputA_set = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(b, 2*simulation_object.feed_size))
        inputB_set = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(b, 2*simulation_object.feed_size))
        
        
        
        t=0
        selected_actions = actions[np.random.randint(0, len(actions), 10)]
        
        
        for j in range(b):
            
            
            input_A = inputA_set[j]
            input_B = inputB_set[j]
            A_t = selected_actions[j]
            
            # get_feedback : phi, psi, user's feedback 값을 구함
            
            psi, s = bench_get_feedback(simulation_object, input_A, input_B, target_w, m="samling")
            psi_d, r_d = get_feedback(A_t, target_w, m="samling")
            
            
            if r_d == -1:
                r_d = 0
            
            
            #true_reward_w[ite].append(get_true_reward(psi, target_w))
            #true_reward_w_d[ite].append(get_true_reward(A_t, target_w))
            #true_reward[ite].append(get_true_reward(opt_trj, target_w))
            
            
            psi_set.append(psi)
            s_set.append(s)
            
            reward_s.append(r_d)
            actions_s.append(A_t)
            
            A_t = A_t.reshape(-1, 1)
            D_PBL.compute_w_t(A_t)
            
            t+=1
        
        
        i = b
        m = 0
        
        
        
        while i < N:
            
            D_PBL.hat_theta_D = fmin_slsqp(regularized_log_likelihood, np.zeros(d),
                                ieqcons=[ieq_const],
                                iprint=0)
            
            w_sampler.A = psi_set
            w_sampler.y = np.array(s_set).reshape(-1,1)
            w_samples = w_sampler.sample(M)

            mean_w_samples = np.mean(w_samples,axis=0)

            if t%(turning_point)==0:
                
                ## change preference parameter (time-varying) ##
                
                print(D_PBL.hat_theta_D)
                print('-------------------------------')
                hat_theta_t.append(D_PBL.hat_theta_D)
                base_theta_t.append(mean_w_samples)
                true_theta_t.append(target_w)
                if tp == 0:
                    target_w = change_w_element(target_w)
                elif tp == 1:
                    target_w = list(np.random.rand(features_d))
                    target_w[0] = np.random.uniform(-0.9,-0.99)
                    target_w[1] = np.random.uniform(-0.9,-0.99)
                    target_w = target_w/np.linalg.norm(target_w)
                
                tp += 1    
                #opt_trj = find_opt_trj(simulation_object, target_w)

            

            
            m = np.dot(mean_w_samples, target_w)/(np.linalg.norm(mean_w_samples)*np.linalg.norm(target_w))
            m_d = np.dot(D_PBL.hat_theta_D, target_w)/(np.linalg.norm(D_PBL.hat_theta_D)*np.linalg.norm(target_w))
            
            
            
            cosine_estimate_w[ite].append(m)
            cosine_estimate_w_d[ite].append(m_d) #robust

            
            
            print('Samples so far: ' + str(i))
            
            # run_algo : query 를 만드는 algorithm
            inputA_set, inputB_set = bench_run_algo(method, simulation_object, w_samples, b, B)
                
            D_PBL.D_rho = D_PBL.D_rho_delta(t)/25
            selected_actions = D_PBL.select_batch_actions(actions, b)
 
            for j in range(b):
                


                
                
                #A_t =D_PBL.select_actions(actions)
                
                
                
                input_A = inputA_set[j] ; input_B = inputB_set[j]
                A_t = selected_actions[j]
                
                psi, s = bench_get_feedback(simulation_object, input_B, input_A, target_w, m="samling")
                psi_d, r_d = get_feedback(A_t, target_w, m="samling")

                if r_d == -1:
                    r_d = 0
                    
                #true_reward_w[ite].append(get_true_reward(psi, target_w))
                #true_reward_w_d[ite].append(get_true_reward(A_t, target_w))
                #true_reward[ite].append(get_true_reward(opt_trj, target_w))
                    
                psi_set.append(psi)
                s_set.append(s)
                reward_s.append(r_d)
                actions_s.append(A_t)
                    
                    
                A_t = A_t.reshape(-1, 1)
                D_PBL.compute_w_t(A_t)
                

                
                    
                t+=1
                    
            i += b
        
        
        D_PBL.hat_theta_D = fmin_slsqp(regularized_log_likelihood, np.zeros(d),
                                        ieqcons=[ieq_const],
                                        iprint=0)
        
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1,1)
        w_samples = w_sampler.sample(M)
        
        
        mean_w_samples = np.mean(w_samples, axis=0)
        
        hat_theta_t.append(D_PBL.hat_theta_D)
        base_theta_t.append(mean_w_samples)
        true_theta_t.append(target_w)

        print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
        m = np.dot(mean_w_samples, target_w)/(np.linalg.norm(mean_w_samples)*np.linalg.norm(target_w))
        m_d = np.dot(D_PBL.hat_theta_D, target_w)/(np.linalg.norm(D_PBL.hat_theta_D)*np.linalg.norm(target_w))
        
        cosine_estimate_w[ite].append(m)
        cosine_estimate_w_d[ite].append(m_d) #robust
        

    hat_theta_t = np.array(hat_theta_t)
    base_theta_t = np.array(base_theta_t)
    
    
    fg = plt.figure(figsize=(10,10))
    
    evaluate_metric = fg.add_subplot(221)
    parameter_circle = fg.add_subplot(222)
    true_reward_metric = fg.add_subplot(223)
    
    
    evaluate_metric.plot(b*np.arange(len(cosine_estimate_w[ite])), np.mean(np.array(cosine_estimate_w), axis=0), color='orange', label='base', alpha=0.8)
    evaluate_metric.plot(b*np.arange(len(cosine_estimate_w_d[ite])), np.mean(np.array(cosine_estimate_w_d), axis=0), color='red', label='D_PBL', alpha=0.8)
    evaluate_metric.set_ylabel('m')
    evaluate_metric.set_xlabel('N')
    evaluate_metric.set_title('cosine metric')
    evaluate_metric.legend()

    draw_circle = plt.Circle((0, 0), 1, fill=False, color='red', zorder=0)
    parameter_circle.add_artist(draw_circle)
    parameter_circle.set_title('Circle')
    parameter_circle.set_xlim([-1.15, 1.15])
    parameter_circle.set_ylim([-1.15, 1.15])

    for i in range(len(hat_theta_t)):
        if i == 0:
            parameter_circle.scatter(true_theta_t[i][0], true_theta_t[i][1], marker='v', zorder=1, color='blue', label='true')
        else:
            parameter_circle.scatter(true_theta_t[i][0], true_theta_t[i][1], marker='v', zorder=1, color='blue')
            
        parameter_circle.annotate(str(i+1), xy=(true_theta_t[i][0],true_theta_t[i][1]),xytext=(10, 10), textcoords='offset pixels')

    parameter_circle.plot(hat_theta_t[:,0], hat_theta_t[:,1], marker='D', zorder=2, color='red', linestyle='dashed', label='D_PBL')
    parameter_circle.plot(base_theta_t[:,0], base_theta_t[:,1], marker='o', zorder=2, color='orange', linestyle='dashed', label='base')
    parameter_circle.legend()
    parameter_circle.set_aspect(1)


    
    true_reward_metric.plot(np.arange(len(true_reward_w[ite])), np.mean(np.array(true_reward_w), axis=0), color='orange', label='base')
    true_reward_metric.plot(np.arange(len(true_reward_w_d[ite])), np.mean(np.array(true_reward_w_d), axis=0), color='red', label='D_PBL')
    true_reward_metric.plot(np.arange(len(true_reward[ite])), np.mean(np.array(true_reward), axis=0), color='blue', label='true', linestyle='dashed')
    
    true_reward_metric.set_ylabel('m')
    true_reward_metric.set_xlabel('N')
    true_reward_metric.set_title('true reward metric')
    true_reward_metric.legend()
    
    plt.savefig('./outputs/robust_time_varying_w_output_1.png')
    plt.show()









def nonbatch(task, method, N, M):
    
    simulation_object = create_env(task)
    d = simulation_object.num_of_features
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

    w_sampler = Sampler(d)
    psi_set = []
    s_set = []
    input_A = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(2*simulation_object.feed_size))
    input_B = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(2*simulation_object.feed_size))
    psi, s = get_feedback(simulation_object, input_A, input_B)
    psi_set.append(psi)
    s_set.append(s)
    for i in range(1, N):
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1,1)
        w_samples = w_sampler.sample(M)
        mean_w_samples = np.mean(w_samples,axis=0)
        print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
        input_A, input_B = run_algo(method, simulation_object, w_samples)
        psi, s = get_feedback(simulation_object, input_A, input_B)
        psi_set.append(psi)
        s_set.append(s)
    w_sampler.A = psi_set
    w_sampler.y = np.array(s_set).reshape(-1,1)
    w_samples = w_sampler.sample(M)
    print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))


