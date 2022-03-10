from sampling import Sampler
import algos
import numpy as np
from simulation_utils import get_feedback, run_algo
import sys
import matplotlib.pyplot as plt
import seaborn as sns


#true_w = [0.29754784,0.03725074,0.00664673,0.80602143]
true_w = list(np.random.rand(4))
estimate_w = [0]

lower_input_bound = -3.14
upper_input_bound = 3.14
#num_of_features = 4
d = 4 # num_of_features



def batch(method, N, M, b):
    
    if N % b != 0:
        print('N must be divisible to b')
        exit(0)
    B = 20*b
    data = np.load('../sampled_trajectories/psi_set.npz')
    data_psi_set = data['PSI_SET']
    # simulation_object = create_env(task)
    # d = simulation_object.num_of_features
    # lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    # upper_input_bound = [x[1] for x in simulation_object.feed_bounds]




    w_sampler = Sampler(d)
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
        
        
        # plot w_samples data
        # 
        # w_samples[:][1]
        # w_samples[:][2]
        # w_samples[:][3]
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
    
    plt.plot(range(len(estimate_w)), estimate_w)
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


