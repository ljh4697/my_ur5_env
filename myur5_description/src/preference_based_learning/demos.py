from sampling import Sampler
import algos
import numpy as np
from simulation_utils import get_feedback, run_algo, get_user_feedback, predict_feedback
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
import copy

#true_w = [0.29754784,0.03725074,0.00664673,0.80602143]
true_w = np.random.rand(4)
true_w = true_w/np.linalg.norm(true_w)

estimate_w = [0]

lower_input_bound = -3.14
upper_input_bound = 3.14
d = 4 # num_of_features


def get_entropy(y, p):
    return -y*np.log(p)-(1-y)*np.log(1-p)






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

def robust_batch(method, N, M, b):
    
    
    e = 5
        
    
    if N % b != 0:
       print('N must be divisible to b')
       exit(0)
    B = 20*b
    data = np.load('../sampled_trajectories/normalized_psi_set.npz')
    data_psi_set = data['PSI_SET']


    estimate_w_o = [[0]for i in range(e)]
    estimate_w = [[0]for i in range(e)]
    #estimate_w_1 = [[0]for i in range(e)]
    #estimate_w_2 = [[0]for i in range(e)]
    estimate_w_r = [[0]for i in range(e)] #robust
    estimate_w_k = [[0]for i in range(e)] #kdpp
    
    corruption_ratio_base = [[0]for i in range(e)]
    corruption_ratio_robust = [[0]for i in range(e)]
    corruption_ratio_kdpp = [[0]for i in range(e)]


    for ite in range(e):
        target_w0 = []
        target_w1 = []
        target_w2 = []
        target_w3 = []
        
        
        #true_w = np.random.rand(4)
        true_w = [np.random.uniform(0,0.1), 0.2, np.random.uniform(0.9,0.99), 0.42]
        true_w = true_w/np.linalg.norm(true_w)
        
        target_w = change_w_element(true_w)
        #target_w = np.random.rand(4)
        #target_w = target_w/np.linalg.norm(target_w)
        
        #target_w=true_w
        t = 1

        w_sampler = Sampler(d)
        oracle_w_sampler = copy.deepcopy(w_sampler)
        #w_sampler_1 = Sampler(d)
        #w_sampler_2 = Sampler(d)
        w_sampler_r = copy.deepcopy(w_sampler)
        w_sampler_k = copy.deepcopy(w_sampler)
        
        
        
        oracle_psi_set = []
        psi_set = []
        psi_set_1 = []
        psi_set_2 = []
        psi_set_r = []
        psi_set_k = []
        
        
        
        oracle_s_set = []
        s_set = []
        s_set_r = []
        
        s_set_1 = []
        s_set_2 = []
        s_set_k = []
        
        t_s_set = []
        t_s_set_r = []
        
        t_s_set_1 = []
        t_s_set_2 = []
        t_s_set_k = []
        
        
        #initialize
        init_psi_id = np.random.randint(0, len(data_psi_set), b)
        
        o_init_psi_id = init_psi_id
        o_init_psi_id_s = init_psi_id
        o_init_psi_id_k = init_psi_id
        
        
        #init_psi_id_1 = np.random.randint(0, len(data_psi_set), int(b/2))
        #init_psi_id_2 = np.random.randint(0, len(data_psi_set), int(b/2))

        
        for j in range(b):
            # get_feedback : phi, psi, user's feedback 값을 구함
            #target_w = get_target_w(true_w, t)
            target_w0.append(target_w[0])
            target_w1.append(target_w[1])
            target_w2.append(target_w[2])
            target_w3.append(target_w[3])
            
            o_psi, _, o_s = get_feedback(data_psi_set[o_init_psi_id[j]], true_w, true_w)
            psi, s, t_s = get_feedback(data_psi_set[init_psi_id[j]], target_w, true_w)
            psi_r, s_r, t_s_r = get_feedback(data_psi_set[o_init_psi_id_s[j]], target_w, true_w)
            psi_k, s_k, t_s_k = get_feedback(data_psi_set[o_init_psi_id_k[j]], target_w, true_w)
            
            
            
            oracle_psi_set.append(o_psi)
            oracle_s_set.append(o_s)
            
            psi_set.append(psi)
            s_set.append(s)

            psi_set_r.append(psi_r)
            s_set_r.append(s_r)
            
            psi_set_k.append(psi_k)
            s_set_k.append(s_k)
            
            t_s_set.append(t_s)
            t_s_set_r.append(t_s_r)
            t_s_set_k.append(t_s_k)

            
            # if j<b/2:
            #     psi_1, s_1, t_s_1 = get_feedback(data_psi_set[init_psi_id_2[j]], target_w, true_w)
            #     psi_2, s_2, t_s_2 = get_feedback(data_psi_set[init_psi_id_1[j]], target_w, true_w)

            #     psi_set_1.append(psi_1)
            #     s_set_1.append(s_1)
                
            #     psi_set_2.append(psi_2)
            #     s_set_2.append(s_2)
                
            #     t_s_set_1.append(t_s_1)
            #     t_s_set_2.append(t_s_2)
            

            t+=1
        i = b
        m = 0
        
        
        while i < N:
            oracle_w_sampler.A = oracle_psi_set
            oracle_w_sampler.y = np.array(oracle_s_set).reshape(-1,1)
            w_samples_o = oracle_w_sampler.sample(M)
            
            w_sampler.A = psi_set
            w_sampler.y = np.array(s_set).reshape(-1,1)
            w_samples = w_sampler.sample(M)
            
            # w_sampler_1.A = psi_set_1
            # w_sampler_1.y = np.array(s_set_1).reshape(-1,1)
            # w_samples_1 = w_sampler_1.sample(M)
            
            # w_sampler_2.A = psi_set_2
            # w_sampler_2.y = np.array(s_set_2).reshape(-1,1)
            # w_samples_2 = w_sampler_2.sample(M)
            
            w_sampler_r.A = psi_set_r
            w_sampler_r.y = np.array(s_set_r).reshape(-1,1)
            w_samples_r = w_sampler_r.sample(M)
            
            w_sampler_k.A = psi_set_k
            w_sampler_k.y = np.array(s_set_k).reshape(-1,1)
            w_samples_k = w_sampler_k.sample(M)
            
            if i%(60)==0 and i <100:
                #target_w = true_w
                target_w = change_w_element(target_w)

            
            
            mean_w_samples_o = np.mean(w_samples_o,axis=0)
            mean_w_samples = np.mean(w_samples,axis=0)
            # mean_w_samples_1 = np.mean(w_samples_1,axis=0)
            # mean_w_samples_2 = np.mean(w_samples_2,axis=0)
            mean_w_samples_r = np.mean(w_samples_r,axis=0)
            mean_w_samples_k = np.mean(w_samples_k,axis=0)
            
            
            current_o_w = mean_w_samples_o/np.linalg.norm(mean_w_samples_o)
            current_w = mean_w_samples/np.linalg.norm(mean_w_samples)
            # current_w_1 = mean_w_samples_1/np.linalg.norm(mean_w_samples_1)
            # current_w_2 = mean_w_samples_2/np.linalg.norm(mean_w_samples_2)
            current_w_s = mean_w_samples_r/np.linalg.norm(mean_w_samples_r)
            current_w_k = mean_w_samples_k/np.linalg.norm(mean_w_samples_k)
            
            
            
            m_o = np.dot(current_o_w, true_w)/(np.linalg.norm(current_o_w)*np.linalg.norm(true_w))
            m = np.dot(current_w, true_w)/(np.linalg.norm(current_w)*np.linalg.norm(true_w))
            # m_1 = np.dot(current_w_1, true_w)/(np.linalg.norm(current_w_1)*np.linalg.norm(true_w))
            # m_2 = np.dot(current_w_2, true_w)/(np.linalg.norm(current_w_2)*np.linalg.norm(true_w))
            m_s = np.dot(current_w_s, true_w)/(np.linalg.norm(current_w_s)*np.linalg.norm(true_w))
            m_k = np.dot(current_w_k, true_w)/(np.linalg.norm(current_w_k)*np.linalg.norm(true_w))
            
            
            
            
            estimate_w_o[ite].append(m_o)
            estimate_w[ite].append(m)
            #estimate_w_1[ite].append(m_1)
            #estimate_w_2[ite].append(m_2)
            estimate_w_r[ite].append(m_s) #robust
            estimate_w_k[ite].append(m_k) # kdpp
            
            

            print('Samples so far: ' + str(i))
            
            
            
            # run_algo :
            psi_set_id_o = run_algo(method, w_samples_o, b, B)
            psi_set_id = run_algo(method, w_samples, b, B)
            # psi_set_id_1 = run_algo('medoids', w_samples_1, int(b/2), B)
            # psi_set_id_2 = run_algo('medoids', w_samples_2, int(b/2), B)
            # psi_set_id_s = run_algo('medoids', w_samples_r, b, B)
            
            if i < N/2:
                #psi_set_id_1 = run_algo('medoids', w_samples_1, int(b/2), B)
                #psi_set_id_2 = run_algo('medoids', w_samples_2, int(b/2), B)
                psi_set_id_s = run_algo('kdpp', w_samples_r, b, B)
                psi_set_id_k = run_algo('kdpp', w_samples_k, b, B)
                
                
            else:
                # psi_set_id_1 = run_algo(method, w_samples_1, int(b/2), B)
                # psi_set_id_2 = run_algo(method, w_samples_2, int(b/2), B)
                psi_set_id_s = run_algo('kdpp', w_samples_r, b, B)
                psi_set_id_k = run_algo(method, w_samples_k, b, B)
                
            corruption_ratio_base[ite].append(len(np.where(np.array(t_s_set) != np.array(s_set))[0])/len(s_set))
            corruption_ratio_robust[ite].append(len(np.where(np.array(t_s_set_r) != np.array(s_set_r))[0])/len(s_set_r))
            corruption_ratio_kdpp[ite].append(len(np.where(np.array(t_s_set_k) != np.array(s_set_k))[0])/len(s_set_k))

            corruption_label_base = []
            corruption_label_robust = []
            corruption_label_kdpp = []
            
            for j in range(b):
        
                #target_w = get_target_w(true_w, t)
                target_w0.append(target_w[0])
                target_w1.append(target_w[1])
                target_w2.append(target_w[2])
                target_w3.append(target_w[3])
                
                o_psi, _, o_s = get_feedback(data_psi_set[psi_set_id_o[j]], true_w, true_w)
                psi, s, t_s = get_feedback(data_psi_set[psi_set_id[j]], target_w, true_w)
                psi_r, s_r, t_s_r = get_feedback(data_psi_set[psi_set_id_s[j]], target_w, true_w)
                psi_k, s_k, t_s_k = get_feedback(data_psi_set[psi_set_id_k[j]], target_w, true_w)
                
                
                oracle_psi_set.append(o_psi)
                oracle_s_set.append(o_s)
                
                psi_set.append(psi)
                s_set.append(s)
                
                psi_set_r.append(psi_r)
                s_set_r.append(s_r)    
                
                psi_set_k.append(psi_k)
                s_set_k.append(s_k)
                
                t_s_set.append(t_s)
                t_s_set_r.append(t_s_r)
                t_s_set_k.append(t_s_k)
                
                
            #     if j<b/2:
                    
            #         psi_1, s_1, t_s_1 = get_feedback(data_psi_set[psi_set_id_2[j]], target_w, true_w)
            #         psi_2, s_2, t_s_2 = get_feedback(data_psi_set[psi_set_id_1[j]], target_w, true_w)



            #         prob1_1 = 1/(1+(np.exp(-s_1*np.dot(current_w_1, data_psi_set[psi_set_id_2[j]].T))))
            #         prob1_2 = 1/(1+(np.exp(-s_1*np.dot(current_w_2, data_psi_set[psi_set_id_1[j]].T))))
                    
            #         #print(f"distance {np.linalg.norm(current_w_1-current_w_2)}")
            #         #print(f"w2 {current_w_2}")
                    
            #         # 1 = prefer A, 0 = prefer B
            #         if s_1 == 1:
            #             z_1 = 1
            #         else:
            #             z_1 = 0
                        
            #         if s_2 == 1:
            #             z_2 = 1
            #         else:
            #             z_2 = 0
                    
            #         # entropy 계산 추가
            #         #print(f"{prob1_1}, {1-prob1_1}")
                    
            #         #sum_of_entropy += np.min((prob1_1, 1-prob1_1))


            #         psi_set_1.append(psi_1)
            #         psi_set_2.append(psi_2)
                     
            #         if i<N/2:
            #             s_set_1.append(s_1)
            #             s_set_2.append(s_2)
            #         else:
            #             s_set_1.append(t_s_1)
            #             s_set_2.append(t_s_2)
                        
                    
            #         t_s_set_1.append(t_s_1)
            #         t_s_set_2.append(t_s_2)
            # last_w1 = mean_w_samples_1
                    
                t+=1
                    
                    


                
                
            i += b
            
        oracle_w_sampler.A = oracle_psi_set
        oracle_w_sampler.y = np.array(oracle_s_set).reshape(-1,1)
        w_samples_o = oracle_w_sampler.sample(M)
            
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1,1)
        w_samples = w_sampler.sample(M)
            
        # w_sampler_1.A = psi_set_1
        # w_sampler_1.y = np.array(s_set_1).reshape(-1,1)
        # w_samples_1 = w_sampler_1.sample(M)
        
        # w_sampler_2.A = psi_set_2
        # w_sampler_2.y = np.array(s_set_2).reshape(-1,1)
        # w_samples_2 = w_sampler_2.sample(M)

        w_sampler_r.A = psi_set_r
        w_sampler_r.y = np.array(s_set_r).reshape(-1,1)
        w_samples_r = w_sampler_r.sample(M)
        
        w_sampler_k.A = psi_set_k
        w_sampler_k.y = np.array(s_set_k).reshape(-1,1)
        w_samples_k = w_sampler_k.sample(M)

        mean_w_samples_o = np.mean(w_samples_o,axis=0)
        mean_w_samples = np.mean(w_samples,axis=0)
        # mean_w_samples_1 = np.mean(w_samples_1,axis=0)
        # mean_w_samples_2 = np.mean(w_samples_2,axis=0)
        mean_w_samples_r = np.mean(w_samples_r,axis=0)
        mean_w_samples_k = np.mean(w_samples_k,axis=0)
        
        
        current_w_o = mean_w_samples_o/np.linalg.norm(mean_w_samples_o)
        current_w = mean_w_samples/np.linalg.norm(mean_w_samples)
        # current_w_1 = mean_w_samples_1/np.linalg.norm(mean_w_samples_1)
        # current_w_2 = mean_w_samples_2/np.linalg.norm(mean_w_samples_2)
        current_w_r = mean_w_samples_r/np.linalg.norm(mean_w_samples_r)
        current_w_k = mean_w_samples_k/np.linalg.norm(mean_w_samples_k)
        
        
        m_o = np.dot(current_w_o, true_w)/(np.linalg.norm(current_w_o)*np.linalg.norm(true_w))
        m = np.dot(current_w, true_w)/(np.linalg.norm(current_w)*np.linalg.norm(true_w))
        # m_1 = np.dot(current_w_1, true_w)/(np.linalg.norm(current_w_1)*np.linalg.norm(true_w))
        # m_2 = np.dot(current_w_2, true_w)/(np.linalg.norm(current_w_2)*np.linalg.norm(true_w))
        m_r = np.dot(current_w_r, true_w)/(np.linalg.norm(current_w_r)*np.linalg.norm(true_w))
        m_k = np.dot(current_w_k, true_w)/(np.linalg.norm(current_w_k)*np.linalg.norm(true_w))
        
        
        estimate_w_o[ite].append(m_o)
        estimate_w[ite].append(m)
        # estimate_w_1[ite].append(m_1)
        # estimate_w_2[ite].append(m_2)
        estimate_w_r[ite].append(m_s) #robust
        estimate_w_k[ite].append(m_k) #kdpp
         
        corruption_ratio_base[ite].append(len(np.where(np.array(t_s_set) != np.array(s_set))[0])/len(s_set))
        corruption_ratio_robust[ite].append(len(np.where(np.array(t_s_set_r) != np.array(s_set_r))[0])/len(s_set_r))        
        corruption_ratio_kdpp[ite].append(len(np.where(np.array(t_s_set_k) != np.array(s_set_k))[0])/len(s_set_k))        
       
        
        #print(selected_ids)
        #print(selected_ids_1)
        #print(selected_ids_2)
        
        # (f"base corruption ratio = {1-(len(np.where(np.array(t_s_set) == np.array(s_set))[0])/len(s_set))}")
        # print(f"model1 corruption ratio = {1-(len(np.where(np.array(t_s_set_1) == np.array(s_set_1))[0])/len(s_set_1))}")
        # print(f"model2 corruption ratio = {1-(len(np.where(np.array(t_s_set_2) == np.array(s_set_2))[0])/len(s_set_2))}")
        
        
        
    
    
    # plot graph
        
    
    fg = plt.figure(figsize=(10,15))
    
    evaluate_metric = fg.add_subplot(321)
    w0 = fg.add_subplot(322)
    w1 = fg.add_subplot(323)
    w2 = fg.add_subplot(324)
    w3 = fg.add_subplot(325)
    corruption_ratio = fg.add_subplot(326)
    
    
    
    evaluate_metric.plot(b*np.arange(len(estimate_w[ite])), np.mean(np.array(estimate_w_o), axis=0), color='blue', label='oracle')
    evaluate_metric.plot(b*np.arange(len(estimate_w[ite])), np.mean(np.array(estimate_w), axis=0), color='violet', label='base')
    #evaluate_metric.plot(b*np.arange(len(estimate_w_1[ite])), np.mean(np.array(estimate_w_1), axis=0), color='green', label='model1')
    #evaluate_metric.plot(b*np.arange(len(estimate_w_1[ite])), np.mean(np.array(estimate_w_2), axis=0), color='orange', label='model2')
    evaluate_metric.plot(b*np.arange(len(estimate_w_r[ite])), np.mean(np.array(estimate_w_r), axis=0), color='red', label='kdpp')
    evaluate_metric.plot(b*np.arange(len(estimate_w_k[ite])), np.mean(np.array(estimate_w_k), axis=0), color='darkcyan', label='kdpp+greedy')
    evaluate_metric.set_ylabel('m')
    evaluate_metric.set_xlabel('N')
    evaluate_metric.set_title('evaluate metric')
    evaluate_metric.legend()
    
        
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
    corruption_ratio.plot(b*np.arange(len(corruption_ratio_base[ite])), np.mean(np.array(corruption_ratio_base), axis=0), color='violet', label='base')
    corruption_ratio.plot(b*np.arange(len(corruption_ratio_base[ite])), np.mean(np.array(corruption_ratio_robust), axis=0), color='red', label='robust')
    corruption_ratio.plot(b*np.arange(len(corruption_ratio_base[ite])), np.mean(np.array(corruption_ratio_kdpp), axis=0), color='darkcyan', label='kdpp')
    corruption_ratio.set_title("corruption_ratio")    
    
    plt.savefig('./outputs/robust_time_varying_w_output_1.png')
    plt.show()
 

def coteaching_batch(method, N, M, b):
    
    e = 1
        
    
    if N % b != 0:
       print('N must be divisible to b')
       exit(0)
    B = 20*b
    data = np.load('../sampled_trajectories/normalized_psi_set.npz')
    data_psi_set = data['PSI_SET']


    estimate_w_o = [[0]for i in range(e)]
    estimate_w = [[0]for i in range(e)]
    estimate_w_1 = [[0]for i in range(e)]
    estimate_w_2 = [[0]for i in range(e)]
    

    


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
        oracle_w_sampler = copy.deepcopy(w_sampler)
        w_sampler_1 = Sampler(d)
        w_sampler_2 = Sampler(d)
        
        
        #sampled w visualization
        # w_samples = w_sampler.sample(M)
        # df = pd.DataFrame(w_samples[:,0])
        # df.plot(kind='density')
        # plt.xlim([-1,1])
        # plt.ylim([0,2])
        # plt.show()
        
        oracle_psi_set = []
        psi_set = []
        psi_set_1 = []
        psi_set_2 = []
        
        
        oracle_s_set = []
        s_set = []
        s_set_1 = []
        s_set_2 = []
        
        t_s_set = []
        t_s_set_1 = []
        t_s_set_2 = []
        
        
        selected_ids = []
        selected_ids_1 = []
        selected_ids_2 = []
        
        #initialize
        init_psi_id = np.random.randint(0, len(data_psi_set), b)
        o_init_psi_id = init_psi_id
    
        init_psi_id_1 = np.random.randint(0, len(data_psi_set), int(b/2))
        init_psi_id_2 = np.random.randint(0, len(data_psi_set), int(b/2))

        selected_ids.append(init_psi_id)
        selected_ids_1.append(init_psi_id_1)
        selected_ids_2.append(init_psi_id_2)
        
        
        for j in range(b):
            # get_feedback : phi, psi, user's feedback 값을 구함
            #target_w = get_target_w(true_w, t)
            target_w0.append(target_w[0])
            target_w1.append(target_w[1])
            target_w2.append(target_w[2])
            target_w3.append(target_w[3])
            
            o_psi, _, o_s = get_feedback(data_psi_set[o_init_psi_id[j]], true_w, true_w)
            psi, s, t_s = get_feedback(data_psi_set[init_psi_id[j]], target_w, true_w)
            
            oracle_psi_set.append(o_psi)
            oracle_s_set.append(o_s)
            
            psi_set.append(psi)
            s_set.append(s)
            t_s_set.append(t_s)

            
            if j<b/2:
                psi_1, s_1, t_s_1 = get_feedback(data_psi_set[init_psi_id_2[j]], target_w, true_w)
                psi_2, s_2, t_s_2 = get_feedback(data_psi_set[init_psi_id_1[j]], target_w, true_w)

                psi_set_1.append(psi_1)
                s_set_1.append(s_1)
                
                psi_set_2.append(psi_2)
                s_set_2.append(s_2)
                
                t_s_set_1.append(t_s_1)
                t_s_set_2.append(t_s_2)
            

            t+=1
        i = b
        m = 0
        
        
        while i < N:
            oracle_w_sampler.A = oracle_psi_set
            oracle_w_sampler.y = np.array(oracle_s_set).reshape(-1,1)
            w_samples_o = oracle_w_sampler.sample(M)
            
            w_sampler.A = psi_set
            w_sampler.y = np.array(s_set).reshape(-1,1)
            w_samples = w_sampler.sample(M)
            
            w_sampler_1.A = psi_set_1
            w_sampler_1.y = np.array(s_set_1).reshape(-1,1)
            w_samples_1 = w_sampler_1.sample(M)
            
            w_sampler_2.A = psi_set_2
            w_sampler_2.y = np.array(s_set_2).reshape(-1,1)
            w_samples_2 = w_sampler_2.sample(M)
            
            if i%(60)==0 and i <100:
              target_w = change_w_element(target_w)
            # if i%(60)==0:
            #     target_w = change_w_element(target_w)
            # if i%240==0:
            #     target_w = change_w_element(target_w)
            
            
            #sampled w visualization
            # df = pd.DataFrame(w_samples[:,0])
            # df.plot(kind='density')
            # plt.xlim([-1,1])
            # plt.ylim([0,2])
            # plt.show()
        
            
            # print(len(w_samples[:,0])) 1000개 w samping
            #input()
            
            print(f'sample length {len(w_samples_1)}')
            print(f'1st w sample {w_samples_1[0]}')
            
            
            mean_w_samples_o = np.mean(w_samples_o,axis=0)
            mean_w_samples = np.mean(w_samples,axis=0)
            mean_w_samples_1 = np.mean(w_samples_1,axis=0)
            mean_w_samples_2 = np.mean(w_samples_2,axis=0)
            
            current_o_w = mean_w_samples_o/np.linalg.norm(mean_w_samples_o)
            current_w = mean_w_samples/np.linalg.norm(mean_w_samples)
            current_w_1 = mean_w_samples_1/np.linalg.norm(mean_w_samples_1)
            current_w_2 = mean_w_samples_2/np.linalg.norm(mean_w_samples_2)
            
            
            m_o = np.dot(current_o_w, true_w)/(np.linalg.norm(current_o_w)*np.linalg.norm(true_w))
            m = np.dot(current_w, true_w)/(np.linalg.norm(current_w)*np.linalg.norm(true_w))
            m_1 = np.dot(current_w_1, true_w)/(np.linalg.norm(current_w_1)*np.linalg.norm(true_w))
            m_2 = np.dot(current_w_2, true_w)/(np.linalg.norm(current_w_2)*np.linalg.norm(true_w))
            
            
            estimate_w_o[ite].append(m_o)
            estimate_w[ite].append(m)
            estimate_w_1[ite].append(m_1)
            estimate_w_2[ite].append(m_2)

            
            
            #print('evaluate metric : {}'.format(m_1))
            #print('w-estimate = {}'.format(current_w_1))
            print('Samples so far: ' + str(i))
            
            
            
            # run_algo :
            psi_set_id_o = run_algo(method, w_samples_o, b, B)
            psi_set_id = run_algo(method, w_samples, b, B)
            
            if i < N/2:
                psi_set_id_1 = run_algo('medoids', w_samples_1, int(b/2), B)
                psi_set_id_2 = run_algo('medoids', w_samples_2, int(b/2), B)
            else:
                psi_set_id_1 = run_algo(method, w_samples_1, int(b/2), B)
                psi_set_id_2 = run_algo(method, w_samples_2, int(b/2), B)
            
            #psi_set_id_1 = algos.optimal_greedy(w_samples_1, int(b/2), i/b, N/2)
            #psi_set_id_2 = algos.optimal_greedy(w_samples_2, int(b/2), i/b, N/2)
            
            
            # psi_set_id_1_o = run_algo("optimal", w_samples_1, 2, B)
            # psi_set_id_2_o = run_algo("optimal", w_samples_2, 2, B)
            
            #print(f'optimal1' + f"{psi_set_id_1_o}")
            # 1, 2 에서 각각 다시 따로 active removal
 
            #selected_ids_o.append(psi_set_id_o)
            selected_ids.append(psi_set_id)
            selected_ids_1.append(psi_set_id_1)
            selected_ids_2.append(psi_set_id_2)

            
            temp_s_1 = []
            temp_s_2 = []
            temp_psi_set_1 = []
            temp_psi_set_2 = []
            
            q_entropy_1 = []
            q_entropy_2 = []
            
            
            for j in range(b):
        
                #target_w = get_target_w(true_w, t)
                target_w0.append(target_w[0])
                target_w1.append(target_w[1])
                target_w2.append(target_w[2])
                target_w3.append(target_w[3])
                
                o_psi, _, o_s = get_feedback(data_psi_set[psi_set_id_o[j]], true_w, true_w)
                psi, s, t_s = get_feedback(data_psi_set[psi_set_id[j]], target_w, true_w)
                
                oracle_psi_set.append(o_psi)
                oracle_s_set.append(o_s)
                
                psi_set.append(psi)
                if i < N/2:
                    s_set.append(s)
                else:
                    s_set.append(t_s)
                t_s_set.append(t_s)

                
                if j<b/2:
                    
                    psi_1, s_1, t_s_1 = get_feedback(data_psi_set[psi_set_id_2[j]], target_w, true_w)
                    psi_2, s_2, t_s_2 = get_feedback(data_psi_set[psi_set_id_1[j]], target_w, true_w)



                    prob1_1 = 1/(1+(np.exp(-s_1*np.dot(current_w_1, data_psi_set[psi_set_id_2[j]].T))))
                    prob1_2 = 1/(1+(np.exp(-s_1*np.dot(current_w_2, data_psi_set[psi_set_id_1[j]].T))))
                    
                    print(f"distance {np.linalg.norm(current_w_1-current_w_2)}")
                    #print(f"w2 {current_w_2}")
                    
                    # 1 = prefer A, 0 = prefer B
                    if s_1 == 1:
                        z_1 = 1
                    else:
                        z_1 = 0
                        
                    if s_2 == 1:
                        z_2 = 1
                    else:
                        z_2 = 0
                    
                    # entropy 계산 추가
                    #print(f"{prob1_1}, {1-prob1_1}")
                    


                    psi_set_1.append(psi_1)
                    psi_set_2.append(psi_2)
                    
                    if i<N/2:
                        s_set_1.append(s_1)
                        s_set_2.append(s_2)
                    else:
                        s_set_1.append(t_s_1)
                        s_set_2.append(t_s_2)
                        
                    
                    t_s_set_1.append(t_s_1)
                    t_s_set_2.append(t_s_2)
                    
                    
            #         q_entropy_1.append(get_entropy(z_1, prob1_1))
            #         q_entropy_2.append(get_entropy(z_2, prob1_2))
                    
            #         temp_psi_set_1.append(psi_1)
            #         temp_s_1.append(s_1)
                    
            #         temp_psi_set_2.append(psi_2)
            #         temp_s_2.append(s_2)
            
            #     t+=1

            # temp_psi_set_1 = np.delete(np.array(temp_psi_set_1), np.argsort(-np.array(q_entropy_1))[:2], axis=0)
            # temp_s_1 = np.delete(np.array(temp_s_1), np.argsort(-np.array(q_entropy_1))[:2])
            # temp_psi_set_2 = np.delete(np.array(temp_psi_set_2), np.argsort(-np.array(q_entropy_2))[:2], axis=0)
            # temp_s_2 = np.delete(np.array(temp_s_2), np.argsort(-np.array(q_entropy_2))[:2])
            
            # print(len(temp_psi_set_1))
            # psi_set_1.extend(list(temp_psi_set_1))
            # s_set_1.extend(list(temp_s_1))
            # print(len(psi_set_1))
            
            
            # psi_set_2.extend(list(temp_psi_set_2))
            # s_set_2.extend(list(temp_s_2))
            # print(len(s_set_2))
                    
                    #t_s_set_1.append(t_s_1)
                    #t_s_set_2.append(t_s_2)
                    
                    
                    
                    
                    
                        
                    # if j<len(psi_set_id_1_o):
                    #     print("update low entropy")
                        
                    #     psi_1_o, s_1_o = get_feedback(data_psi_set[psi_set_id_2_o[j]], target_w)
                    #     psi_2_o, s_2_o = get_feedback(data_psi_set[psi_set_id_1_o[j]], target_w)

                    #     psi_set_1.append(psi_1_o)
                    #     s_set_1.append(s_1_o)
                        
                    #     psi_set_2.append(psi_2_o)
                    #     s_set_2.append(s_2_o)
                    

                    
                
                
            i += b
            
        oracle_w_sampler.A = oracle_psi_set
        oracle_w_sampler.y = np.array(oracle_s_set).reshape(-1,1)
        w_samples_o = oracle_w_sampler.sample(M)
            
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1,1)
        w_samples = w_sampler.sample(M)
            
        w_sampler_1.A = psi_set_1
        w_sampler_1.y = np.array(s_set_1).reshape(-1,1)
        w_samples_1 = w_sampler_1.sample(M)
        
        w_sampler_2.A = psi_set_2
        w_sampler_2.y = np.array(s_set_2).reshape(-1,1)
        w_samples_2 = w_sampler_2.sample(M)


        mean_w_samples_o = np.mean(w_samples_o,axis=0)
        mean_w_samples = np.mean(w_samples,axis=0)
        mean_w_samples_1 = np.mean(w_samples_1,axis=0)
        mean_w_samples_2 = np.mean(w_samples_2,axis=0)
        
        current_w_o = mean_w_samples_o/np.linalg.norm(mean_w_samples_o)
        current_w = mean_w_samples/np.linalg.norm(mean_w_samples)
        current_w_1 = mean_w_samples_1/np.linalg.norm(mean_w_samples_1)
        current_w_2 = mean_w_samples_2/np.linalg.norm(mean_w_samples_2)
        
        m_o = np.dot(current_w_o, true_w)/(np.linalg.norm(current_w_o)*np.linalg.norm(true_w))
        m = np.dot(current_w, true_w)/(np.linalg.norm(current_w)*np.linalg.norm(true_w))
        m_1 = np.dot(current_w_1, true_w)/(np.linalg.norm(current_w_1)*np.linalg.norm(true_w))
        m_2 = np.dot(current_w_2, true_w)/(np.linalg.norm(current_w_2)*np.linalg.norm(true_w))
        
        
        estimate_w_o[ite].append(m_o)
        estimate_w[ite].append(m)
        estimate_w_1[ite].append(m_1)
        estimate_w_2[ite].append(m_2)
        
        
        #print(selected_ids)
        #print(selected_ids_1)
        #print(selected_ids_2)
        
        print(f"base corruption ratio = {1-(len(np.where(np.array(t_s_set) == np.array(s_set))[0])/len(s_set))}")
        print(f"model1 corruption ratio = {1-(len(np.where(np.array(t_s_set_1) == np.array(s_set_1))[0])/len(s_set_1))}")
        print(f"model2 corruption ratio = {1-(len(np.where(np.array(t_s_set_2) == np.array(s_set_2))[0])/len(s_set_2))}")
        
        print(len(np.where(np.array(t_s_set) == np.array(s_set))[0]))
        
        
        
        count_query = {}
        count_query_1 = {}
        count_query_2 = {}
        
        for ids in selected_ids:
            for id in ids:
                if id not in count_query:
                    count_query[id] = 1
                else:
                    count_query[id]+=1
                    
        for ids in selected_ids_1:
            for id in ids:
                if id not in count_query_1:
                    count_query_1[id] = 1
                else:
                    count_query_1[id]+=1
                    
        for ids in selected_ids_2:
            for id in ids:
                if id not in count_query_2:
                    count_query_2[id] = 1
                else:
                    count_query_2[id]+=1
        
        
        
    
    fg_0 = plt.figure(figsize=(5,15))
    
    c = fg_0.add_subplot(311)
    c1 = fg_0.add_subplot(312)
    c2 = fg_0.add_subplot(313)
    
    c.bar(count_query.keys(), count_query.values())
    c1.bar(count_query_1.keys(), count_query_1.values())
    c2.bar(count_query_2.keys(), count_query_2.values())
    
    
    # plot graph
        
    
    fg = plt.figure(figsize=(10,15))
    
    evaluate_metric = fg.add_subplot(321)
    w0 = fg.add_subplot(322)
    w1 = fg.add_subplot(323)
    w2 = fg.add_subplot(324)
    w3 = fg.add_subplot(325)
    
    
    
    #plt.subplot(2, 2, 1)
    evaluate_metric.plot(b*np.arange(len(estimate_w[ite])), np.mean(np.array(estimate_w_o), axis=0), color='blue', label='oracle')
    evaluate_metric.plot(b*np.arange(len(estimate_w[ite])), np.mean(np.array(estimate_w), axis=0), color='violet', label='base')
    evaluate_metric.plot(b*np.arange(len(estimate_w_1[ite])), np.mean(np.array(estimate_w_1), axis=0), color='green', label='model1')
    evaluate_metric.plot(b*np.arange(len(estimate_w_1[ite])), np.mean(np.array(estimate_w_2), axis=0), color='orange', label='model2')
    evaluate_metric.set_ylabel('m')
    evaluate_metric.set_xlabel('N')
    evaluate_metric.set_title('evaluate metric')
    evaluate_metric.legend()
    
    
        
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

    plt.savefig('./outputs/robust_time_varying_w_output_1.png')
    plt.show()
    
    
def batch(method, N, M, b):
    # stack_batch_active
    
    e = 1
        
    
    if N % b != 0:
        print('N must be divisible to b')
        exit(0)
    B = 20*b
    data = np.load('../sampled_trajectories/psi_set.npz')
    data_psi_set = data['PSI_SET']

    estimate_w = [[0]for i in range(e)]
    stack_estimate_w = [[0]for i in range(e)]


    for ite in range(e):
        target_w0 = []
        target_w1 = []
        target_w2 = []
        target_w3 = []
        
        
        true_w = np.random.rand(4)
        true_w = true_w/np.linalg.norm(true_w)
        
        #target_w=true_w
        target_w = change_w_element(true_w)
        t = 1


        w_sampler = Sampler(d)
        w_samples = w_sampler.sample(M)
        mean_w_samples = np.mean(w_samples,axis=0)
        
        stack_W_sampler = Sampler(d)
        
        #sampled w visualization
        # w_samples = w_sampler.sample(M)
        # df = pd.DataFrame(w_samples[:,0])
        # df.plot(kind='density')
        # plt.xlim([-1,1])
        # plt.ylim([0,2])
        # plt.show()
        
        psi_set = []
        stack_psi_set = []
        
        s_set = []
        stack_s_set = []
        
        predicted_s_set = []
        groundtruth_s_set = []
        feedback_entropy_arxive = []
        
        stack = {}
        
        
        
        init_psi_id = np.random.randint(0, len(data_psi_set), b)
        
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
            
            stack_psi_set.append(psi)
            stack_s_set.append(s)
            
            
            t+=1
            
        
        
        
        i = b
        m = 0
        
        
        while i < N:
            
            w_sampler.A = psi_set
            w_sampler.y = np.array(s_set).reshape(-1,1)
            w_samples = w_sampler.sample(M)
            mean_w_samples = np.mean(w_samples,axis=0)
            current_w = mean_w_samples/np.linalg.norm(mean_w_samples)
            m = np.dot(current_w, true_w)/(np.linalg.norm(current_w)*np.linalg.norm(true_w))
            estimate_w[ite].append(m)
            
            
            stack_W_sampler.A = stack_psi_set
            stack_W_sampler.y = np.array(stack_s_set).reshape(-1,1)
            stack_w_samples = stack_W_sampler.sample(M)
            
            stack_mean_w_samples = np.mean(stack_w_samples,axis=0)
            stack_current_w = stack_mean_w_samples/np.linalg.norm(stack_mean_w_samples)
            
            stack_m = np.dot(stack_current_w, true_w)/(np.linalg.norm(stack_current_w)*np.linalg.norm(stack_current_w))
            stack_estimate_w[ite].append(stack_m)
            
            feedback_entropy, predicted_s= predict_feedback(stack_psi_set, stack_mean_w_samples)
            
            
            # predicted label != target label 인경우 stack+1, stack이 s넘으면 pseudo label로 교체
            for idx in np.where(np.array(stack_s_set)*np.array(predicted_s)==-1)[0]:
                if not idx in stack:
                    stack[idx]=1
                else:
                    stack[idx]+=1
                    if stack[idx]>=int(N/b):
                        stack_s_set[idx] = predicted_s[idx]
                            
                        
                        
            
            feedback_entropy_arxive.append(feedback_entropy)
            predicted_s_set.append(predicted_s)



            #sampled w visualization
            # df = pd.DataFrame(w_samples[:,0])
            # df.plot(kind='density')
            # plt.xlim([-1,1])  
            # plt.ylim([0,2])
            # plt.show()
        
            
            # print(len(w_samples[:,0])) 1000개 w samping
            #input()
            
            #print(f'sample length {len(w_samples)}')
            #print(f'1st w sample {w_samples[0]}')
            
            if i%(50)==0:
                target_w = change_w_element(target_w)
            if i%200==0:
                target_w = change_w_element(target_w)
                

            
            
            #print('evaluate metric : {}'.format(m))
            #print('w-estimate = {}'.format(current_w))
            print('Samples so far: ' + str(i))
            
            # run_algo :
            psi_set_id = run_algo(method, w_samples, b, B)
            stack_psi_set_id = run_algo(method, stack_w_samples, b, B)
            
            for j in range(b):
        
                #target_w = get_target_w(true_w, t)
                target_w0.append(target_w[0])
                target_w1.append(target_w[1])
                target_w2.append(target_w[2])
                target_w3.append(target_w[3])
                
                psi, s = get_feedback(data_psi_set[psi_set_id[j]], target_w)

                psi_set.append(psi)
                s_set.append(s)

                stack_psi, stack_s = get_feedback(data_psi_set[stack_psi_set_id[j]], target_w)

                stack_psi_set.append(stack_psi)
                stack_s_set.append(stack_s)
                
                t+=1
                

            
                
                
            i += b
        
        
        # w* 의 ground truth label 계산
        _, groundtruth_s_set = predict_feedback(psi_set, true_w)
            
            
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1,1)
        w_samples = w_sampler.sample(M)
        
        mean_w_samples = np.mean(w_samples,axis=0)
        current_w = mean_w_samples/np.linalg.norm(mean_w_samples)
        
        m = np.dot(current_w, true_w)/(np.linalg.norm(current_w)*np.linalg.norm(true_w))
        estimate_w[ite].append(m)
        
        
        stack_W_sampler.A = stack_psi_set
        stack_W_sampler.y = np.array(stack_s_set).reshape(-1,1)
        stack_w_samples = stack_W_sampler.sample(M)
        
        stack_mean_w_samples = np.mean(stack_w_samples,axis=0)
        stack_current_w = stack_mean_w_samples/np.linalg.norm(stack_mean_w_samples)
        
        stack_m = np.dot(stack_current_w, true_w)/(np.linalg.norm(stack_current_w)*np.linalg.norm(stack_current_w))
        stack_estimate_w[ite].append(stack_m)
        
        
        
        feedback_entropy, predicted_s= predict_feedback(stack_psi_set, stack_mean_w_samples)
        
        feedback_entropy_arxive.append(feedback_entropy)
        predicted_s_set.append(predicted_s)

        
        


        #print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
        print('Samples so far: ' + str(i))
        
        
    p_t_label_set = []
    
    correct_ratio_ = []
    wrong_label = []
    diff_p_t_label = []
    argmax_entropy = []
    count_stack = {}
    
    # target label을 predict label 로 대체했을 때 true label과 일치율
    correct_change_ratio = []
    
    # target label vs groundtruth label
    
    # predict label 과 target label
    
    # predict label 과 target label 이 같으면 1 아니면 -1
    for p_s_set in predicted_s_set:
        p_t_label_set.append(list(map(lambda x,y:x*y, p_s_set,s_set[:len(p_s_set)])))
        
    t_correct_label_set = np.array(groundtruth_s_set) * np.array(s_set)
    
    # target != ground truth 인 label (training시 잘 못 labeling 해준 것.)
    wrong_label = np.where(np.array(t_correct_label_set) == -1)[0]
    
    
    # iteration 별로 predicted != target 인 label
    for correct_label in p_t_label_set:
        correct_ratio_.append(correct_label.count(1)/(len(correct_label)))
        # 매 iter 별로 corre
        diff_p_t_label.append(np.where(np.array(correct_label)==-1))
        print(np.where(np.array(correct_label)==-1)[0])
        for idx in np.where(np.array(correct_label)==-1)[0]:
            if not idx in count_stack:
                count_stack[idx]=1
            else:
                count_stack[idx]+=1
            
    
    #entropy 가 가장 높은 2개 query의 argument
    for x in feedback_entropy_arxive:
        argmax_entropy.append(np.argsort(-np.array(x))[:2])
        
    
    #print(p_t_label_set)
    #print('############')
    #print(t_correct_label_set)
    print('#################')
    print(diff_p_t_label)
    print('##################')
    print(wrong_label)
    print('##################')
    print(count_stack)
    
    #print(feedback_entropy_arxive)
    x=np.arange(len(count_stack))
    
    val = []
    colors = []
    for s in sorted(count_stack.keys()):
       val.append(count_stack[s]) 
       if s in wrong_label:
           colors.append('r')
       else:
           colors.append('b')
           
    
    plt.bar(x, val, color=colors)
    plt.xticks(x, sorted(count_stack.keys()))
    plt.show()
    
        
    
    fg = plt.figure(figsize=(10,15))
    
    evaluate_metric = fg.add_subplot(321)
    w0 = fg.add_subplot(322)
    w1 = fg.add_subplot(323)
    w2 = fg.add_subplot(324)
    w3 = fg.add_subplot(325)
    correct_ratio = fg.add_subplot(326)

    
    
    
    #plt.subplot(2, 2, 1)
    evaluate_metric.plot(b*np.arange(len(estimate_w[ite])), np.mean(np.array(estimate_w), axis=0), color='violet', label='base')
    evaluate_metric.plot(b*np.arange(len(stack_estimate_w[ite])), np.mean(np.array(stack_estimate_w), axis=0), color='orange', label='stack')
    evaluate_metric.set_ylabel('m')
    evaluate_metric.set_xlabel('N')
    evaluate_metric.set_title('evaluate metric')
    evaluate_metric.legend()
    
        
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
    
    correct_ratio.plot(10+b*np.arange(len(correct_ratio_)), correct_ratio_)
    correct_ratio.set_xlabel('N')
    correct_ratio.set_ylabel('correct ratio')
    correct_ratio.set_title('tcorrect_ratio')

    
    
    plt.savefig('./outputs/stationary_w_output_2.png')
    plt.show()
    



def o_batch(method, N, M, b):
    
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


