from sampling import Sampler
import algos
import numpy as np
from simulation_utils import get_feedback, run_algo, get_user_feedback
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import copy
from test_mesh_pickandplace import create_environment
import control_planning_scene
import scipy.optimize as opt
import algos
from scipy.stats import kde
import pandas as pd

#true_w = [0.29754784,0.03725074,0.00664673,0.80602143]
true_w = np.random.rand(4)
true_w = true_w/np.linalg.norm(true_w)

estimate_w = [0]

lower_input_bound = -3.14
upper_input_bound = 3.14
d = 4 # num_of_features

def adversarial_batch(method, N, M, b):
    
    
    
    e = 1
    
    def get_target_w(true_w, t):
        n_w = copy.deepcopy(true_w)
        
        n_w[0] += (1.2/np.sqrt(t))*np.sin(t/4)
        n_w[1] += (1/np.sqrt(t))*np.sin(t/3)
        
        #n_w[1] -= (2/t)*np.cos(t)
        #n_w[2] += (1/t)*np.sin(t)
        #n_w[3] += (3/t)*np.cos(t)
        
        n_w = n_w/np.linalg.norm(n_w)
        return n_w
        
        
    def change_w_element(true_w):
        
        
        n_w = copy.deepcopy(true_w)
        
        max_id = np.argmax(n_w)
        min_id = np.argmin(n_w)
        
        
        
        max_v = n_w[max_id]
        min_v = n_w[min_id]
        
        n_w[max_id] = min_v
        n_w[min_id] = max_v
        
        
        return n_w

    
    
    if N % b != 0:
        print('N must be divisible to b')
        exit(0)
    B = 20*b
    data = np.load('../sampled_trajectories/psi_set.npz')
    data_psi_set = data['PSI_SET']

    estimate_w = [[0]for i in range(e)]
    






    for ite in range(e):
        target_w0 = []
        target_w1 = []
        target_w2 = []
        target_w3 = []
        
        
        true_w = np.random.rand(4)
        true_w = true_w/np.linalg.norm(true_w)
        
        target_w = change_w_element(true_w)
        #target_w=true_w
        t = 1


        w_sampler = Sampler(d)
        
        #sampled w visualization
        # w_samples = w_sampler.sample(M)
        # df = pd.DataFrame(w_samples[:,0])
        # df.plot(kind='density')
        # plt.xlim([-1,1])
        # plt.ylim([0,2])
        # plt.show()
        
        psi_set = []
        s_set = []
         
        
        
        init_psi_id = np.random.randint(1, 100, b)
        
        for j in range(b):
            # get_feedback : phi, psi, user's feedback 값을 구함
            #target_w = get_target_w(true_w, t)
            target_w0.append(target_w[0])
            target_w1.append(target_w[1])
            target_w2.append(target_w[2])
            target_w3.append(target_w[3])
            
            
            psi, s = get_feedback(data_psi_set[init_psi_id[j]], target_w)

            psi_set.append(psi)
            s_set.append(s)
            t+=1
        i = b
        m = 0
        
        
        while i < N:
            w_sampler.A = psi_set
            w_sampler.y = np.array(s_set).reshape(-1,1)
            w_samples = w_sampler.sample(M)
            
            if i == N/2:
                target_w = change_w_element(target_w)
            #sampled w visualization
            # df = pd.DataFrame(w_samples[:,0])
            # df.plot(kind='density')
            # plt.xlim([-1,1])
            # plt.ylim([0,2])
            # plt.show()
        
            
            # print(len(w_samples[:,0])) 1000개 w samping
            #input()
            
            print(f'sample length {len(w_samples)}')
            print(f'1st w sample {w_samples[0]}')
            
            
            mean_w_samples = np.mean(w_samples,axis=0)
            current_w = mean_w_samples/np.linalg.norm(mean_w_samples)
            
            m = np.dot(current_w, true_w)/(np.linalg.norm(current_w)*np.linalg.norm(true_w))
            estimate_w[ite].append(m)
            
            
            print('evaluate metric : {}'.format(m))
            print('w-estimate = {}'.format(current_w))
            print('Samples so far: ' + str(i))
            
            # run_algo :
            psi_set_id = run_algo(method, w_samples, b, B)
            for j in range(b):
        
                #target_w = get_target_w(true_w, t)
                target_w0.append(target_w[0])
                target_w1.append(target_w[1])
                target_w2.append(target_w[2])
                target_w3.append(target_w[3])
                
                psi, s = get_feedback(data_psi_set[psi_set_id[j]], target_w)

                psi_set.append(psi)
                s_set.append(s)
                
                t+=1
                
                
            i += b
            
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1,1)
        w_samples = w_sampler.sample(M)
        mean_w_samples = np.mean(w_samples, axis=0)
        print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
        
    
    fg = plt.figure(figsize=(10,18))
    
    evaluate_metric = fg.add_subplot(321)
    w0 = fg.add_subplot(322)
    w1 = fg.add_subplot(323)
    w2 = fg.add_subplot(324)
    w3 = fg.add_subplot(325)
    
    
    
    #plt.subplot(2, 2, 1)
    evaluate_metric.plot(b*np.arange(len(estimate_w[ite])), np.mean(np.array(estimate_w), axis=0))
    evaluate_metric.set_ylabel('m')
    evaluate_metric.set_xlabel('N')
    evaluate_metric.set_title('evaluate metric')
    
        
    w0.plot(np.arange(N), target_w0)
    w0.plot(np.arange(N), np.ones(N)*true_w[0], 'r--')
    w0.set_xlabel('N')
    w0.set_ylabel('w0')
    w0.set_title('target w0')
    
    w1.plot(np.arange(N), target_w1)
    w1.plot(np.arange(N), np.ones(N)*true_w[1], 'r--')
    w1.set_xlabel('N')
    w1.set_ylabel('w1')
    w1.set_title('target w1')

    w2.plot(np.arange(N), target_w2)
    w2.plot(np.arange(N), np.ones(N)*true_w[2], 'r--')
    w2.set_xlabel('N')
    w2.set_ylabel('w2')
    w2.set_title('target w2')
    
    w3.plot(np.arange(N), target_w3)
    w3.plot(np.arange(N), np.ones(N)*true_w[3], 'r--')
    w3.set_xlabel('N')
    w3.set_ylabel('w3')
    w3.set_title('target w3')
    

    target_w1
    
    plt.savefig('./outputs/targetw_output_ch_2.png')
    plt.show()
    



def batch(method, N, M, b):
    
    if N % b != 0:
        print('N must be divisible to b')
        exit(0)
    B = 20*b
    data = np.load('../sampled_trajectories/psi_set.npz')
    data_psi_set = data['PSI_SET']


    w_sampler = Sampler(d)
    
    #sampled w visualization
    # w_samples = w_sampler.sample(M)
    # df = pd.DataFrame(w_samples[:,0])
    # df.plot(kind='density')
    # plt.xlim([-1,1])
    # plt.ylim([0,2])
    # plt.show()
    
    psi_set = []
    s_set = []
    
    
    
    init_psi_id = np.random.randint(1, 100, b)
    
    for j in range(b):
        # get_feedback : phi, psi, user's feedback 값을 구함
        psi, s = get_feedback(data_psi_set[init_psi_id[j]], true_w)

        psi_set.append(psi)
        s_set.append(s)
    i = b
    m = 0
    
    
    while i < N:
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1,1)
        w_samples = w_sampler.sample(M)
        
        #sampled w visualization
        # df = pd.DataFrame(w_samples[:,0])
        # df.plot(kind='density')
        # plt.xlim([-1,1])
        # plt.ylim([0,2])
        # plt.show()

        
            
        
        print(len(w_samples[:,0]))
        #input()
        
        print(f'sample length {len(w_samples)}')
        print(f'1st w sample {w_samples[0]}')
        
        
        mean_w_samples = np.mean(w_samples,axis=0)
        current_w = mean_w_samples/np.linalg.norm(mean_w_samples)
        
        m = np.dot(current_w, true_w)/(np.linalg.norm(current_w)*np.linalg.norm(true_w))
        estimate_w.append(m)
        
        
        print('evaluate metric : {}'.format(m))
        print('w-estimate = {}'.format(current_w))
        print('Samples so far: ' + str(i))
        
        # run_algo :
        psi_set_id = run_algo(method, w_samples, b, B)
        for j in range(b):
            
            
            psi, s = get_feedback(data_psi_set[psi_set_id[j]], true_w)

            psi_set.append(psi)
            s_set.append(s)
            
            
        i += b
        
    w_sampler.A = psi_set
    w_sampler.y = np.array(s_set).reshape(-1,1)
    w_samples = w_sampler.sample(M)
    mean_w_samples = np.mean(w_samples, axis=0)
    print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
    
    plt.plot(10*np.arange(len(estimate_w)), estimate_w)
    plt.ylabel('m')
    plt.xlabel('N')
    plt.savefig('./outputs/output.png')
    plt.show()




def user_batch(method, N, M, b):
    
    if N % b != 0:
        print('N must be divisible to b')
        exit(0)
    B = 20*b
    data = np.load('../sampled_trajectories/psi_set.npz')
    data_psi_set = data['PSI_SET']


    w_sampler = Sampler(d)
    psi_set = []
    s_set = []
    


    # make env from mujoco world
    planning_scene_1 = control_planning_scene.control_planning_scene()

    env, grasp_point, approach_direction, objects_co, neutral_position = create_environment(planning_scene_1)

    planning_scene_1.remove_object(objects_co['milk'])
    planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
    


    #initialize psi id
    init_psi_id = np.random.randint(1, 100, b)
    



    for j in range(b):
        # get_feedback : phi, psi, user's feedback 값을 구함
        if j == 0:
            print("1st query for batch")
        elif j == 1:
            print("2nd query for batch")
        elif j == 2:
            print("3rd query for batch")
        else:
            print(f"{j+1}th query for batch")

        psi, s = get_user_feedback(data_psi_set[init_psi_id[j]], init_psi_id[j], objects_co)


        psi_set.append(psi)
        s_set.append(s)

    i = b
    m = 0 # evaluate metric
    
    
    while i < N:
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1,1)
        w_samples = w_sampler.sample(M)
        
        

        print(len(w_samples[:,0]))
        #input()
        
        print(f'sample length {len(w_samples)}')
        print(f'1st w sample {w_samples[0]}')
        
        
        mean_w_samples = np.mean(w_samples,axis=0)
        current_w = mean_w_samples/np.linalg.norm(mean_w_samples)
        
        m = np.dot(current_w, true_w)/(np.linalg.norm(current_w)*np.linalg.norm(true_w))
        estimate_w.append(m)
        
        
        print('evaluate metric : {}'.format(m))
        print('w-estimate = {}'.format(current_w))
        print('Samples so far: ' + str(i))
        
        # run_algo :
        psi_set_id = run_algo(method, w_samples, b, B)
        for j in range(b):

            if j == 0:
                print("1st query for batch")
            elif j == 1:
                print("2nd query for batch")
            elif j == 2:
                print("3rd query for batch")
            else:
                print(f"{j+1}th query for batch")
            
            psi, s = get_user_feedback(data_psi_set[init_psi_id[j]], init_psi_id[j], objects_co)

            psi_set.append(psi)
            s_set.append(s)
            
            
        i += b
        
    w_sampler.A = psi_set
    w_sampler.y = np.array(s_set).reshape(-1,1)
    w_samples = w_sampler.sample(M)
    mean_w_samples = np.mean(w_samples, axis=0)
    print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
    
    plt.plot(10*np.arange(len(estimate_w)), estimate_w)
    plt.ylabel('m')
    plt.xlabel('N')
    plt.savefig('./outputs/output.png')
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


