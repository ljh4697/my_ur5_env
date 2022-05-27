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


# batch 에서 적은 entropy select 추가하기 전


#true_w = [0.29754784,0.03725074,0.00664673,0.80602143]
true_w = np.random.rand(4)
true_w = true_w/np.linalg.norm(true_w)

estimate_w = [0]

lower_input_bound = -3.14
upper_input_bound = 3.14
d = 3 # num_of_features




def get_point_data():
    
    fname_data = '/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/ctrl_samples/preference_point.csv'
    data        = np.genfromtxt(fname_data, delimiter=',')
    label   = data[:, 2]

    number_data = data.shape[0]
    
    b_data = np.ones((number_data, 3))

    point_x = data[:, 0]
    point_y = data[:, 1]
    label   = data[:, 2]
    
    b_data[:, 1] = point_x ; b_data[:, 2] = point_y

    point_class_0 = np.zeros((len(point_x[label == 0]), 2))
    point_class_0[:,0] = point_x[label == 0]
    point_class_0[:,1] = point_y[label == 0]

    point_class_1 = np.zeros((len(point_x[label == 1]), 2))
    point_class_1[:,0] = point_x[label == 1]
    point_class_1[:,1] = point_y[label == 1]
    
    
    # psi data가 point data 라고 가정

    b_data[:, 1] = point_x
    b_data[:, 2] = point_y
    
    
    # plt.figure(figsize=(8,8))   
    # plt.title('training data')
    # plt.plot(point_class_0[:,0], point_class_0[:,1], 'o', color='lightblue', label='class = 0')
    # plt.plot(point_class_1[:,0], point_class_1[:,1], 'o', color='lightgreen', label='class = 1')
    # plt.axis('equal')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
        
    
    
    
    return b_data, label, point_class_0, point_class_1 

def compute_linear_regression(theta, point):
    
    
    value = np.dot(point, theta.T)
    
    return value


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



def coteaching_batch():
    
    
    method = 'greedy'
    N = 140
    M = 1000
    b = 20
    e = 1
    
    if N % b != 0:
       print('N must be divisible to b')
       exit(0)
    B = 20*b
    
    data_psi_set, label, point_class_0, point_class_1 = get_point_data()



    estimate_w_o = [[0]for i in range(e)]
    estimate_w = [[0]for i in range(e)]
    estimate_w_1 = [[0]for i in range(e)]
    estimate_w_2 = [[0]for i in range(e)]
    estimate_w_s = [[0]for i in range(e)]
    corruption_ratio_base = [[0]for i in range(e)]
    corruption_ratio_robust = [[0]for i in range(e)]
    
    # mesh grid
    X = np.arange(-20, 35, 0.1) # USE THIS VALUE for the range of x values in the construction of coordinate
    Y = np.arange(-20, 35, 0.1) # USE THIS VALUE for the range of y values in the construction of coordinate

    [XX, YY] = np.meshgrid(X, Y)


    m_point = np.ones((XX.size,3))
    m_point[:, 1] = XX.reshape(-1)
    m_point[:, 2] = YY.reshape(-1)
    


    for ite in range(e):
        target_w0 = []
        target_w1 = []
        target_w2 = []
        
        t = 1
        true_w = np.array([-0.02809329, 0.36071584, -0.4335038 ])
        
        
        #target_w = change_w_element(true_w)
        target_w=np.array([0.04, 0.21584, -0.7835038  ])

        w_sampler = Sampler(d)
        oracle_w_sampler = copy.deepcopy(w_sampler)
        #w_sampler_1 = Sampler(d)
        #w_sampler_2 = Sampler(d)
        w_sampler_s = copy.deepcopy(w_sampler)
        
        
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
        psi_set_s = []
        
        
        oracle_s_set = []
        s_set = []
        s_set_1 = []
        s_set_2 = []
        s_set_s = []
        
        t_s_set = []
        t_s_set_1 = []
        t_s_set_2 = []
        t_s_set_s = []
        
        

        
        #initialize
        init_psi_id = np.random.randint(0, len(data_psi_set), b)
        o_init_psi_id = init_psi_id
        o_init_psi_id_s = init_psi_id
    
        init_psi_id_1 = np.random.randint(0, len(data_psi_set), int(b/2))
        init_psi_id_2 = np.random.randint(0, len(data_psi_set), int(b/2))
        
        
        for j in range(b):
            # get_feedback : phi, psi, user's feedback 값을 구함
            #target_w = get_target_w(true_w, t)
            target_w0.append(target_w[0])
            target_w1.append(target_w[1])
            target_w2.append(target_w[2])
            
            o_psi, _, o_s = get_feedback(data_psi_set[o_init_psi_id[j]], true_w, true_w)
            psi, s, t_s = get_feedback(data_psi_set[init_psi_id[j]], target_w, true_w)
            psi_s, s_s, t_s_s = get_feedback(data_psi_set[o_init_psi_id_s[j]], target_w, true_w)
            
            oracle_psi_set.append(o_psi)
            oracle_s_set.append(o_s)
            
            psi_set.append(psi)
            s_set.append(s)
            t_s_set.append(t_s)
            t_s_set_s.append(t_s_s)

            psi_set_s.append(psi_s)
            s_set_s.append(s_s)
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
        corruption_id_base = []
        corruption_id_robust = []
        
        
        
        while i < N:
            oracle_w_sampler.A = oracle_psi_set
            oracle_w_sampler.y = np.array(oracle_s_set).reshape(-1,1)
            w_samples_o = oracle_w_sampler.sample(M)
            
            w_sampler.A = psi_set
            w_sampler.y = np.array(s_set).reshape(-1,1)
            w_samples = w_sampler.sample(M)
            
            
            w_sampler_s.A = psi_set_s
            w_sampler_s.y = np.array(s_set_s).reshape(-1,1)
            w_samples_s = w_sampler_s.sample(M)
            
            # w_sampler_1.A = psi_set_1
            # w_sampler_1.y = np.array(s_set_1).reshape(-1,1)
            # w_samples_1 = w_sampler_1.sample(M)
            
            # w_sampler_2.A = psi_set_2
            # w_sampler_2.y = np.array(s_set_2).reshape(-1,1)
            # w_samples_2 = w_sampler_2.sample(M)

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
        
            
            if i%(60)==0 and i <100:
                target_w = copy.deepcopy(true_w)
                

            
            print(f"target_w {target_w}")
            
            mean_w_samples_o = np.mean(w_samples_o,axis=0)
            mean_w_samples = np.mean(w_samples,axis=0)
            # mean_w_samples_1 = np.mean(w_samples_1,axis=0)
            # mean_w_samples_2 = np.mean(w_samples_2,axis=0)
            mean_w_samples_s = np.mean(w_samples_s,axis=0)
            
            current_o_w = mean_w_samples_o/np.linalg.norm(mean_w_samples_o)
            current_w = mean_w_samples/np.linalg.norm(mean_w_samples)
            # current_w_1 = mean_w_samples_1/np.linalg.norm(mean_w_samples_1)
            # current_w_2 = mean_w_samples_2/np.linalg.norm(mean_w_samples_2)
            current_w_s = mean_w_samples_s/np.linalg.norm(mean_w_samples_s)
            
            
            m_o = np.dot(current_o_w, true_w)/(np.linalg.norm(current_o_w)*np.linalg.norm(true_w))
            m = np.dot(current_w, true_w)/(np.linalg.norm(current_w)*np.linalg.norm(true_w))
            # m_1 = np.dot(current_w_1, true_w)/(np.linalg.norm(current_w_1)*np.linalg.norm(true_w))
            # m_2 = np.dot(current_w_2, true_w)/(np.linalg.norm(current_w_2)*np.linalg.norm(true_w))
            m_s = np.dot(current_w_s, true_w)/(np.linalg.norm(current_w_s)*np.linalg.norm(true_w))
            
            
            estimate_w_o[ite].append(m_o)
            estimate_w[ite].append(m)
            estimate_w_s[ite].append(m_s)

            
            
            #print('evaluate metric : {}'.format(m_1))
            #print('w-estimate = {}'.format(current_w_1))
            print('Samples so far: ' + str(i))
            
            
            # run_algo :
            psi_set_id_o = algos.point_greedy(w_samples_o, b, data_psi_set)
            psi_set_id = algos.point_greedy(w_samples, b, data_psi_set)
            m_psi_set_id = algos.point_medoids(w_samples, b, data_psi_set)
            
            #psi_set_id_1 = run_algo(method, w_samples_1, int(b/2), B)
            #psi_set_id_2 = run_algo(method, w_samples_2, int(b/2), B)
            
            if i < N/2:
                psi_set_id_s = algos.point_medoids(w_samples_s, b, data_psi_set)
            else:
                psi_set_id_s = algos.point_greedy(w_samples_s, b, data_psi_set)
                
                
            
            
            
            d_bdry_s = np.where(np.round(compute_linear_regression(current_w_s, m_point),2)==0)
            d_bdry = np.where(np.round(compute_linear_regression(current_w, m_point),2)==0)
            target_d_bdry = np.where(np.round(compute_linear_regression(target_w, m_point),2)==0)
            true_d_bdry = np.where(np.round(compute_linear_regression(true_w, m_point),2)==0)
            
            

            
            
            corruption_ratio_base[ite].append(len(np.where(np.array(t_s_set) != np.array(s_set))[0])/len(s_set))
            corruption_ratio_robust[ite].append(len(np.where(np.array(t_s_set_s) != np.array(s_set_s))[0])/len(s_set_s))

            
            
            
            

                

            # 1, 2 에서 각각 다시 따로 active removal
 
            #selected_ids_o.append(psi_set_id_o)

            
            corruption_label_base = []
            corruption_label_robust = []
            
            
            for j in range(b):
        
                #target_w = get_target_w(true_w, t)
                target_w0.append(target_w[0])
                target_w1.append(target_w[1])
                target_w2.append(target_w[2])
                
                o_psi, _, o_s = get_feedback(data_psi_set[psi_set_id_o[j]], true_w, true_w)
                psi, s, t_s = get_feedback(data_psi_set[psi_set_id[j]], target_w, true_w)
                psi_s, s_s, t_s_s = get_feedback(data_psi_set[psi_set_id_s[j]], target_w, true_w)
                
                if s != t_s:
                    corruption_label_base.append(j)
                    
                if s_s != t_s_s:
                    corruption_label_robust.append(j)
                    
                    
                                    
                oracle_psi_set.append(o_psi)
                oracle_s_set.append(o_s)
                
                psi_set.append(psi)
                s_set.append(s)
                t_s_set.append(t_s)
                t_s_set_s.append(t_s_s)
                
                

                psi_set_s.append(psi_s)
                s_set_s.append(s_s)
                
                # if j<b/2:
                    
                #     psi_1, s_1, t_s_1 = get_feedback(data_psi_set[psi_set_id_2[j]], target_w, true_w)
                #     psi_2, s_2, t_s_2 = get_feedback(data_psi_set[psi_set_id_1[j]], target_w, true_w)



                #     prob1_1 = 1/(1+(np.exp(-s_1*np.dot(current_w_1, data_psi_set[psi_set_id_2[j]].T))))
                #     prob1_2 = 1/(1+(np.exp(-s_1*np.dot(current_w_2, data_psi_set[psi_set_id_2[j]].T))))
                    
                    
                    
                #     # entropy 계산 추가
                #     print(t_s_1)
                #     print(f"{prob1_1}, {1-prob1_1}")
                    
                    
                    
                    
                #     temp_psi_set_1.append(psi_1)
                #     temp_s_1.append(s_1)
                    
                #     temp_psi_set_2.append(psi_2)
                #     temp_s_2.append(s_2)
            



                #     psi_set_1.append(psi_1)
                #     s_set_1.append(s_1)
                    
                #     psi_set_2.append(psi_2)
                #     s_set_2.append(s_2)
                    
                #     t_s_set_1.append(t_s_1)
                #     t_s_set_2.append(t_s_2)
                    
                    
                    
                    
                    
                        
                    # if j<len(psi_set_id_1_o):
                    #     print("update low entropy")
                        
                    #     psi_1_o, s_1_o = get_feedback(data_psi_set[psi_set_id_2_o[j]], target_w)
                    #     psi_2_o, s_2_o = get_feedback(data_psi_set[psi_set_id_1_o[j]], target_w)

                    #     psi_set_1.append(psi_1_o)
                    #     s_set_1.append(s_1_o)
                        
                    #     psi_set_2.append(psi_2_o)
                    #     s_set_2.append(s_2_o)
                    

                    
                t+=1
                
            
            ## 이전에 corruption 된 label 과 같은것을 뽑앗을 때 corruption label 로 지정
            if len(corruption_label_base):
                corruption_id_base = np.append(corruption_id_base, psi_set_id[np.array(corruption_label_base)])
                
                
            corruption_check_base = np.where(np.array(list(map(lambda x: x in corruption_id_base, psi_set_id)))==1)[0]
            #corrupted_base = psi_set_id[corruption_check_base]
                #psi_set_id = np.delete(psi_set_id, np.array(corruption_label_base))
            print(f"corruption_base {psi_set_id[corruption_check_base]}")
            
            if len(corruption_label_robust):
                corruption_id_robust = np.append(corruption_id_robust, psi_set_id_s[np.array(corruption_label_robust)])
                #psi_set_id_s = np.delete(psi_set_id_s, np.array(corruption_label_robust))
            
            corruption_check_robust = np.where(np.array(list(map(lambda x: x in corruption_id_robust, psi_set_id_s)))==1)[0]
            
            print(f"corruption_robust {psi_set_id_s[corruption_check_robust]}")
            
                
            plt.figure(figsize=(16,8))   
            
            plt.subplot(1,2,1)
            plt.title('greedy selection')
            plt.plot(point_class_0[:,0], point_class_0[:,1], 'o', color='lightblue', label='class = 0')
            plt.plot(point_class_1[:,0], point_class_1[:,1], 'o', color='lightgreen', label='class = 1')
            plt.plot(m_point[:, 1][d_bdry], m_point[:, 2][d_bdry], color='violet')
            plt.plot(m_point[:, 1][d_bdry_s], m_point[:, 2][d_bdry_s], color='red')
            plt.plot(m_point[:, 1][target_d_bdry], m_point[:, 2][target_d_bdry],'-.', color='darkslategray')
            plt.plot(m_point[:, 1][true_d_bdry], m_point[:, 2][true_d_bdry], '-.', color='blue')
            plt.plot(data_psi_set[psi_set_id,1], data_psi_set[psi_set_id,2], 'o', color='purple', label='query')
            if len(corruption_check_base):
                plt.plot(data_psi_set[psi_set_id[corruption_check_base],1], data_psi_set[psi_set_id[corruption_check_base],2], 'o', color='orange', label='corruption')
            
            plt.axis('equal')
            plt.legend()
            plt.tight_layout()
            
            plt.subplot(1,2,2)
            plt.title('robust selection')
            plt.plot(point_class_0[:,0], point_class_0[:,1], 'o', color='lightblue', label='class = 0')
            plt.plot(point_class_1[:,0], point_class_1[:,1], 'o', color='lightgreen', label='class = 1')
            plt.plot(m_point[:, 1][d_bdry], m_point[:, 2][d_bdry], color='violet')
            plt.plot(m_point[:, 1][d_bdry_s], m_point[:, 2][d_bdry_s], color='red')
            plt.plot(m_point[:, 1][target_d_bdry], m_point[:, 2][target_d_bdry], '-.',color='darkslategray')
            plt.plot(m_point[:, 1][true_d_bdry], m_point[:, 2][true_d_bdry], '-.',color='blue')
            plt.plot(data_psi_set[psi_set_id_s,1], data_psi_set[psi_set_id_s,2], 'o', color='purple', label='query')
            if len(corruption_check_robust):
                plt.plot(data_psi_set[psi_set_id_s[corruption_check_robust],1], data_psi_set[psi_set_id_s[corruption_check_robust],2], 'o', color='orange', label='corruption')
            plt.axis('equal')
            plt.legend()
            plt.tight_layout()
            
            plt.show()
                
            q = input()
            if q == "q":
                exit()
                
                
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

        w_sampler_s.A = psi_set_s
        w_sampler_s.y = np.array(s_set_s).reshape(-1,1)
        w_samples_s = w_sampler_s.sample(M)

        mean_w_samples_o = np.mean(w_samples_o,axis=0)
        mean_w_samples = np.mean(w_samples,axis=0)
        # mean_w_samples_1 = np.mean(w_samples_1,axis=0)
        # mean_w_samples_2 = np.mean(w_samples_2,axis=0)
        mean_w_samples_s = np.mean(w_samples_s,axis=0)
        
        current_w_o = mean_w_samples_o/np.linalg.norm(mean_w_samples_o)
        current_w = mean_w_samples/np.linalg.norm(mean_w_samples)
        # current_w_1 = mean_w_samples_1/np.linalg.norm(mean_w_samples_1)
        # current_w_2 = mean_w_samples_2/np.linalg.norm(mean_w_samples_2)
        current_w_s = mean_w_samples_s/np.linalg.norm(mean_w_samples_s)
        
        m_o = np.dot(current_w_o, true_w)/(np.linalg.norm(current_w_o)*np.linalg.norm(true_w))
        m = np.dot(current_w, true_w)/(np.linalg.norm(current_w)*np.linalg.norm(true_w))
        # m_1 = np.dot(current_w_1, true_w)/(np.linalg.norm(current_w_1)*np.linalg.norm(true_w))
        # m_2 = np.dot(current_w_2, true_w)/(np.linalg.norm(current_w_2)*np.linalg.norm(true_w))
        m_s = np.dot(current_w_s, true_w)/(np.linalg.norm(current_w_s)*np.linalg.norm(true_w))
        
        
        estimate_w_o[ite].append(m_o)
        estimate_w[ite].append(m)
        # estimate_w_1[ite].append(m_1)
        # estimate_w_2[ite].append(m_2)
        estimate_w_s[ite].append(m_s)
        corruption_ratio_base[ite].append(len(np.where(np.array(t_s_set) != np.array(s_set))[0])/len(s_set))
        corruption_ratio_robust[ite].append(len(np.where(np.array(t_s_set_s) != np.array(s_set_s))[0])/len(s_set_s))        

        
        #print(f"base corruption ratio = {1-(len(np.where(np.array(t_s_set) == np.array(s_set))[0])/len(s_set))}")
        # print(f"model1 corruption ratio = {1-(len(np.where(np.array(t_s_set_1) == np.array(s_set_1))[0])/len(s_set_1))}")
        # print(f"model2 corruption ratio = {1-(len(np.where(np.array(t_s_set_2) == np.array(s_set_2))[0])/len(s_set_2))}")
        
        
        

    # plot graph
        
    
    fg = plt.figure(figsize=(10,15))
    
    evaluate_metric = fg.add_subplot(321)
    w0 = fg.add_subplot(322)
    w1 = fg.add_subplot(323)
    w2 = fg.add_subplot(324)
    corruption_ratio = fg.add_subplot(325)
    
    
    
    
    
    #plt.subplot(2, 2, 1)
    evaluate_metric.plot(b*np.arange(len(estimate_w[ite])), np.mean(np.array(estimate_w_o), axis=0), color='blue', label='oracle')
    evaluate_metric.plot(b*np.arange(len(estimate_w[ite])), np.mean(np.array(estimate_w), axis=0), color='violet', label='base')
    #evaluate_metric.plot(b*np.arange(len(estimate_w_1[ite])), np.mean(np.array(estimate_w_1), axis=0), color='green', label='model1')
    #evaluate_metric.plot(b*np.arange(len(estimate_w_1[ite])), np.mean(np.array(estimate_w_2), axis=0), color='orange', label='model2')
    evaluate_metric.plot(b*np.arange(len(estimate_w_s[ite])), np.mean(np.array(estimate_w_s), axis=0), color='red', label='robust')
    
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
    
    corruption_ratio.plot(b*np.arange(len(corruption_ratio_base[ite])), np.mean(np.array(corruption_ratio_base), axis=0), color='violet', label='base')
    corruption_ratio.plot(b*np.arange(len(corruption_ratio_base[ite])), np.mean(np.array(corruption_ratio_robust), axis=0), color='red', label='robust')
    evaluate_metric.set_xlabel('N')
    evaluate_metric.legend()
    corruption_ratio.set_title("corruption_ratio")    
    
    #plt.savefig('./outputs/robust_time_varying_w_output_1.png')
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    coteaching_batch()
    
