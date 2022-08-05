from sampling import Sampler
import bench_algos
import numpy as np
from simulation_utils import create_env, get_feedback, run_algo, bench_run_algo, bench_get_feedback
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import copy
#from bandit_base import GLUCB
from scipy.optimize import fmin_slsqp
from run_optimizer import get_opt_f, get_opt_feature, generate_features

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

def bench_experiment(task, method, N, b, DPB_params, batch_active_params , num_randomseeds=1):
    np.random.seed(num_randomseeds)
    
    
    
    ###########  set hyperparams  #############################
    ###########################################################
    ''' DPB_params '''
    
    alpha = DPB_params["exploration_weight"] 

    
    ''' batch_active_params'''
    
    M = batch_active_params["samples_num"]
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ##############################################################
    
    turning_point = int(N/3) # param change point



    def get_true_reward(action, param_star):
        return np.dot(action,param_star)
    
    # e = num_randomseeds
    B = 20*b
    
    if N % b != 0:
        print('N must be divisible to b')
        exit(0)
    
    
    
    '''evaluate metric arxive'''
    cosine_estimate_w = [[0]for i in range(e)]
    cosine_estimate_w_d = [[0]for i in range(e)] #
    
    true_reward_w = [[0]for i in range(e)]
    true_reward_w_d = [[0]for i in range(e)] #
    true_reward = [[0]for i in range(e)]
    
    
    
    
    
    simulation_object = create_env(task)
    
    
    
    ##### get psi set ######
    data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/ctrl_samples/' + simulation_object.name + '.npz')
    actions = data['psi_set']
    #trajectory_set = data['inputs_set']
    features_d = simulation_object.num_of_features
    d = features_d
    
    features_data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/ctrl_samples/' + simulation_object.name + '_features'+'.npz')
    predefined_features = features_data['features']
    
    
    
    tp = 0 # count change # of true param
    t=0 # count iteration
    
    
    s_set = []
    
    psi_set = []
    s_set = []
    
    hat_theta_t = []
    base_theta_t = []
    true_theta_t = []

    # initialize theta_t^star
    
    true_w = list(np.random.rand(features_d))
    true_w[0] = np.random.uniform(0,0.1)
    true_w[1] = np.random.uniform(0.9,0.99)
    true_w[2] = 0.3
    true_w[3] = 0.2
    
    true_w = true_w/np.linalg.norm(true_w)
    target_w = change_w_element(true_w)
    
    
    DPB = GLUCB(d=features_d, DPB_params=DPB_params)
    
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

    w_sampler = Sampler(d)
    
    
    


    # random select psi (initialization)
    inputA_set = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(b, 2*simulation_object.feed_size))
    inputB_set = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(b, 2*simulation_object.feed_size))
    selected_actions = actions[np.random.randint(0, len(actions), 10)]
    
    
    for j in range(b):
        
        
        input_A = inputA_set[j]
        input_B = inputB_set[j]
        A_t = selected_actions[j]
        
        # get user's feedback(label) 값을 구함
        psi, s = bench_get_feedback(simulation_object, input_A, input_B, target_w, m="samling")
        psi_d, r_d = get_feedback(A_t, target_w, m="samling")
        
        
        if r_d == -1:
            r_d = 0

        
        psi_set.append(psi)
        s_set.append(s)
        
        DPB.reward_s.append(r_d)
        DPB.actions_s.append(A_t)
        
        A_t = A_t.reshape(-1, 1)
        DPB.compute_w_t(A_t)
        
        t+=1
    
    
    i = b
    m_b = 0
    
    
    
    while i < N:
        
        
        # update param #
        DPB.update_param(t)
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1,1)
        w_samples = w_sampler.sample(M)

        mean_w_samples = np.mean(w_samples,axis=0)




        if t%(turning_point)==0:
            ## change preference parameter (time-varying) ##
            
            print(DPB.hat_theta_D)
            print('-------------------------------')
            hat_theta_t.append(copy.deepcopy(DPB.hat_theta_D))
            base_theta_t.append(copy.deepcopy(mean_w_samples))
            true_theta_t.append(copy.deepcopy(target_w))
            if tp == 0:
                target_w = change_w_element(target_w)
            elif tp == 1:
                target_w = list(np.random.rand(features_d))
                target_w[0] = np.random.uniform(-0.9,-0.99)
                target_w[1] = np.random.uniform(-0.9,-0.99)
                target_w = target_w/np.linalg.norm(target_w)
            
            tp += 1    
            
            #opt_trj = find_opt_trj(simulation_object, target_w)

        

        # cosine metric m_b #
        m_b = np.dot(mean_w_samples, target_w)/(np.linalg.norm(mean_w_samples)*np.linalg.norm(target_w))
        m_d = np.dot(DPB.hat_theta_D, target_w)/(np.linalg.norm(DPB.hat_theta_D)*np.linalg.norm(target_w))
        
        cosine_estimate_w[ite].append(m_b)
        cosine_estimate_w_d[ite].append(m_d) #robust
        
        #true rewawrd metric#
        base_opt_feature = get_opt_f(predefined_features, mean_w_samples)
        D_opt_feature = get_opt_f(predefined_features, DPB.hat_theta_D)
        true_opt_feature = get_opt_f(predefined_features, target_w)  

        true_reward_w[ite].append(np.dot(target_w, base_opt_feature))
        true_reward_w_d[ite].append(np.dot(target_w, D_opt_feature))
        true_reward[ite].append(np.dot(target_w, true_opt_feature))
        
        
        
        print('Samples so far: ' + str(i))
        
        
        # select psi (arm) by each algorithm
        inputA_set, inputB_set = bench_run_algo(method, simulation_object, w_samples, b, B)
        
        DPB.D_rho = DPB.D_rho_delta(t)*alpha
        selected_actions = DPB.select_batch_actions(actions, b)
        for j in range(b):
            
            
            #A_t =DPB.select_actions(actions)
            
            input_A = inputA_set[j] ; input_B = inputB_set[j]
            A_t = selected_actions[j]
            
            psi, s = bench_get_feedback(simulation_object, input_B, input_A, target_w, m="samling")
            psi_d, r_d = get_feedback(A_t, target_w, m="samling")

            if r_d == -1:
                r_d = 0

                
            psi_set.append(psi)
            s_set.append(s)
            DPB.reward_s.append(r_d)
            DPB.actions_s.append(A_t)
                
                
            A_t = A_t.reshape(-1, 1)
            DPB.compute_w_t(A_t)
            

                
            t+=1
                
        i += b
        
        
        
        # update param #
        
        DPB.update_param(t)
        
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1,1)
        w_samples = w_sampler.sample(M)
        
        
        mean_w_samples = np.mean(w_samples, axis=0)
        
        hat_theta_t.append(copy.deepcopy(DPB.hat_theta_D))
        base_theta_t.append(copy.deepcopy(mean_w_samples))
        true_theta_t.append(copy.deepcopy(target_w))

        # cosine similarity metric = m_b
        print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
        m_b = np.dot(mean_w_samples, target_w)/(np.linalg.norm(mean_w_samples)*np.linalg.norm(target_w))
        m_d = np.dot(DPB.hat_theta_D, target_w)/(np.linalg.norm(DPB.hat_theta_D)*np.linalg.norm(target_w))
        cosine_estimate_w[ite].append(m_b)
        cosine_estimate_w_d[ite].append(m_d)
        
        # true rewawrd metric
        base_opt_feature = get_opt_f(predefined_features, mean_w_samples)
        D_opt_feature = get_opt_f(predefined_features, DPB.hat_theta_D)
        true_opt_feature = get_opt_f(predefined_features, target_w)  

        true_reward_w[ite].append(np.dot(target_w, base_opt_feature))
        true_reward_w_d[ite].append(np.dot(target_w, D_opt_feature))
        true_reward[ite].append(np.dot(target_w, true_opt_feature))






    ################################
    # plot experiment result graph #
    ################################

    hat_theta_t = np.array(hat_theta_t)
    base_theta_t = np.array(base_theta_t)
    
    
    fg = plt.figure(figsize=(10,10))
    
    evaluate_metric = fg.add_subplot(221)
    parameter_circle = fg.add_subplot(222)
    true_reward_metric = fg.add_subplot(223)
    
    
    evaluate_metric.plot(b*np.arange(len(cosine_estimate_w[ite])), np.mean(np.array(cosine_estimate_w), axis=0), color='orange', label='base', alpha=0.8)
    evaluate_metric.plot(b*np.arange(len(cosine_estimate_w_d[ite])), np.mean(np.array(cosine_estimate_w_d), axis=0), color='red', label='DPB', alpha=0.8)
    evaluate_metric.set_ylabel('cosine_m')
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

    parameter_circle.plot(hat_theta_t[:,0], hat_theta_t[:,1], marker='D', zorder=2, color='red', linestyle='dashed', label='DPB')
    parameter_circle.plot(base_theta_t[:,0], base_theta_t[:,1], marker='o', zorder=2, color='orange', linestyle='dashed', label='base')
    parameter_circle.legend()
    parameter_circle.set_aspect(1)


    
    true_reward_metric.plot(b*np.arange(len(true_reward_w[ite])), np.mean(np.array(true_reward_w), axis=0), color='orange', label='base')
    true_reward_metric.plot(b*np.arange(len(true_reward_w_d[ite])), np.mean(np.array(true_reward_w_d), axis=0), color='red', label='DPB')
    true_reward_metric.plot(b*np.arange(len(true_reward[ite])), np.mean(np.array(true_reward), axis=0), color='blue', label='true', linestyle='dashed')
    
    true_reward_metric.set_ylabel('simple_regret')
    true_reward_metric.set_xlabel('N')
    true_reward_metric.set_title('true reward metric')
    true_reward_metric.legend()
    
    plt.savefig('./outputs/' + str(N) +'_alpha_' + str(alpha) +"_gamma_" + str(DPB_params["gamma"]) +'.png')
    #plt.show()









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


