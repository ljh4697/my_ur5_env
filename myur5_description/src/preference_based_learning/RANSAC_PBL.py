from sampling import Sampler
import algos
import numpy as np
#from simulation_utils import get_feedback, run_algo, get_user_feedback, predict_feedback
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import copy
from simulation_utril_origin import get_feedback
from test_mesh_pickandplace import create_environment
import control_planning_scene
import scipy.optimize as opt
import algos
from scipy.stats import kde
import pandas as pd
import copy
from scipy.stats import gaussian_kde
from tqdm import trange
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
    # plt.plot(point_class_0[:,0], point_class_0[:,1], 'o', color='lightblue')
    # plt.plot(point_class_1[:,0], point_class_1[:,1], 'o', color='lightblue')
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

def RANSAC(alpha, psi_set, label_set, n, tau):
    
    num_of_set = len(label_set)
    Outlierscore = np.zeros(num_of_set)
    
    for i in trange(n):
        ransac_sampler = Sampler(d)
        selected_id = np.random.choice(num_of_set, int(alpha*num_of_set), replace=False)
        ransac_sampler.A = psi_set[selected_id]
        ransac_sampler.y = label_set[selected_id]
        #print(selected_id)
        ransac_w_samples=ransac_sampler.sample(1000)
        mean_w_samples_rnsc = np.mean(ransac_w_samples,axis=0)
        t_r = psi_set@mean_w_samples_rnsc
        #print(t_r)
        t_r[np.where(t_r>=0)]=1
        t_r[np.where(t_r<0)]=-1
        #print(t_r)
        #print(label_set.reshape(-1))
        #print(np.where(np.array(label_set.reshape(-1))!=t_r)[0])
        Outlierscore[np.where(np.array(label_set.reshape(-1))!=t_r)[0]]+=1
        
        ########################
        
    Inlier_set=np.where(Outlierscore<=tau)[0]
    Inlier_sampler = Sampler(d)
    Inlier_sampler.A = psi_set[Inlier_set]
    Inlier_sampler.y = label_set[Inlier_set]
    Inlier_w_samples=Inlier_sampler.sample(1000)
    mean_w_samples_Inlier = np.mean(Inlier_w_samples,axis=0)
    
    t_r = psi_set@mean_w_samples_Inlier
    t_r[np.where(t_r>=0)]=1
    t_r[np.where(t_r<0)]=-1
    Final_psi_id = np.array(list(set(Inlier_set).union(set(np.where(np.array(label_set).reshape(-1)==t_r)[0]))))
    Final_sampler = Sampler(d)
    Final_sampler.A = psi_set[Final_psi_id]
    Final_sampler.y = label_set[Final_psi_id]
    FInall_w_samples=Final_sampler.sample(1000)
    mean_w_samples_Final = np.mean(FInall_w_samples,axis=0)
    
    return mean_w_samples_Final
    
    
        
    
        
    

def RANSAC_PBL_Toy():
    
    method = 'greedy'
    N = 100
    M = 1000
    b = 10
    e = 1
    if N % b != 0:
       print('N must be divisible to b')
       exit(0)
    B = 20*b
    data_psi_set, label, point_class_0, point_class_1 = get_point_data()
    estimate_w_r = [[0]for i in range(e)] #robust
    estimate_w = [[0]for i in range(e)] #origin
    estimate_w_o = [[0]for i in range(e)]
    
    
    # # mesh grid
    # X = np.arange(-20, 35, 0.1) # USE THIS VALUE for the range of x values in the construction of coordinate
    # Y = np.arange(-20, 35, 0.1) # USE THIS VALUE for the range of y values in the construction of coordinate
    # [XX, YY] = np.meshgrid(X, Y)
    # m_point = np.ones((XX.size,3))
    # m_point[:, 1] = XX.reshape(-1)
    # m_point[:, 2] = YY.reshape(-1)
    
    
    for ite in range(e):
        target_w0 = []
        target_w1 = []
        target_w2 = []
        t = 1
        true_w = np.array([-0.02809329, 0.36071584, -0.4335038 ])
        #target_w = change_w_element(true_w)
        target_w=np.array([0.04, 0.231584, -0.835038  ]) #ex1
        #target_w=np.array([-4.4036, 0.2404, 0.2  ]) # perpendicular
        w_sampler = Sampler(d)
        oracle_w_sampler = copy.deepcopy(w_sampler)
        
        w_sampler_r = copy.deepcopy(w_sampler)
        
        w_samples = w_sampler.sample(M)
        w_samples_r = w_sampler_r.sample(M)
        
        
        
        # # Sampling graph plot
        # fg0 = plt.figure(figsize=(10,5))
        # ax0 = fg0.add_subplot(121)
        # ax1 = fg0.add_subplot(122)
        # xy0 = np.vstack([w_samples[:,1],w_samples[:,2]])
        # z0 = gaussian_kde(xy0)(xy0)
        # idx0 = z0.argsort()
        # x0, y0, z0 = w_samples[:,1][idx0], w_samples[:,2][idx0], z0[idx0]
        # ax0.set_title('greedy')
        # ax0.scatter(x0, y0, c=z0, s=10, cmap='bone')
        # ax0.scatter(target_w[1],target_w[2], s=100, c='orange')
        # ax0.set_xlim([-1,1])
        # ax0.set_ylim([-1,1])
        # ax0.set_xlabel('w1')
        # ax0.set_ylabel('w2')
        # xy1 = np.vstack([w_samples_k[:,1],w_samples_k[:,2]])
        # z1 = gaussian_kde(xy1)(xy1)
        # idx1 = z1.argsort()
        # x1, y1, z1 = w_samples_k[:,1][idx1], w_samples_k[:,2][idx1], z1[idx1]
        # ax1.set_title('diverse')
        # ax1.scatter(x1, y1, c=z1, s=10, cmap='bone')
        # ax1.scatter(target_w[1],target_w[2], s=100, c='orange')
        # ax1.set_xlim([-1,1])
        # ax1.set_ylim([-1,1])
        # ax1.set_xlabel('w1')
        # ax1.set_ylabel('w2')
        # plt.tight_layout()
        # plt.show()
        
        oracle_psi_set = []
        psi_set = []
        psi_set_r = []
        
        oracle_s_set = []
        s_set = []
        o_s_set = []
        
        s_set_r = []
        
        #initialize
        init_psi_id = np.random.randint(0, len(data_psi_set), b)
        init_psi_id_o = init_psi_id
        init_psi_id_r = init_psi_id 
        
        for j in range(b):
            # get_feedback : phi, psi, user's feedback 값을 구함
            #target_w = get_target_w(true_w, t)
            target_w0.append(target_w[0])
            target_w1.append(target_w[1])
            target_w2.append(target_w[2])
            
            o_psi, _, o_s = get_feedback(data_psi_set[init_psi_id_o[j]], true_w, true_w)
            psi, s, t_s = get_feedback(data_psi_set[init_psi_id[j]], target_w, true_w)
            psi_r, s_r, t_s_r = get_feedback(data_psi_set[init_psi_id_r[j]], target_w, true_w)
            
            
            oracle_psi_set.append(o_psi)
            oracle_s_set.append(o_s)
            
            psi_set.append(psi)
            s_set.append(s)
            o_s_set.append(t_s)

            psi_set_r.append(psi_r)
            s_set_r.append(s_r)
            
            
        i = b
        m = 0 
        
        while i < N:
            oracle_w_sampler.A = oracle_psi_set
            oracle_w_sampler.y = np.array(oracle_s_set).reshape(-1,1)
            w_samples_o = oracle_w_sampler.sample(M)
            
            w_sampler.A = psi_set
            w_sampler.y = np.array(s_set).reshape(-1,1)
            w_samples = w_sampler.sample(M)
            
            
            w_sampler_r.A = psi_set_r
            w_sampler_r.y = np.array(s_set_r).reshape(-1,1)
            w_samples_r = w_sampler_r.sample(M)
            
            
            
            if i%(40)==0 and i <100:
                target_w = copy.deepcopy(true_w)


            
            # fg0 = plt.figure(figsize=(10,5))
            # ax0 = fg0.add_subplot(121)
            # ax1 = fg0.add_subplot(122)
            
            
            # xy0 = np.vstack([w_samples[:,1],w_samples[:,2]])
            # z0 = gaussian_kde(xy0)(xy0)
            # idx0 = z0.argsort()
            # x0, y0, z0 = w_samples[:,1][idx0], w_samples[:,2][idx0], z0[idx0]

            # ax0.set_title('greedy')
            # ax0.scatter(x0, y0, c=z0, s=10, cmap='bone')
            # ax0.scatter(target_w[1],target_w[2], s=100, c='orange')
            # ax0.set_xlim([-1,1])
            # ax0.set_ylim([-1,1])
            # ax0.set_xlabel('w1')
            # ax0.set_ylabel('w2')
            

            
            # plt.tight_layout()
            
            # plt.show()
            
            
            
            print(f"target_w {target_w}")
            
            mean_w_samples_o = np.mean(w_samples_o,axis=0)
            mean_w_samples = np.mean(w_samples,axis=0)
            # mean_w_samples_1 = np.mean(w_samples_1,axis=0)
            # mean_w_samples_2 = np.mean(w_samples_2,axis=0)
            mean_w_samples_r = np.mean(w_samples_r,axis=0)
            
            
            current_o_w = mean_w_samples_o/np.linalg.norm(mean_w_samples_o)
            current_w = mean_w_samples/np.linalg.norm(mean_w_samples)
            # current_w_1 = mean_w_samples_1/np.linalg.norm(mean_w_samples_1)
            # current_w_2 = mean_w_samples_2/np.linalg.norm(mean_w_samples_2)
            current_w_r = mean_w_samples_r/np.linalg.norm(mean_w_samples_r)
            
            
            
            m_o = np.dot(current_o_w, true_w)/(np.linalg.norm(current_o_w)*np.linalg.norm(true_w))
            m = np.dot(current_w, true_w)/(np.linalg.norm(current_w)*np.linalg.norm(true_w))
            # m_1 = np.dot(current_w_1, true_w)/(np.linalg.norm(current_w_1)*np.linalg.norm(true_w))
            # m_2 = np.dot(current_w_2, true_w)/(np.linalg.norm(current_w_2)*np.linalg.norm(true_w))
            m_r = np.dot(current_w_r, true_w)/(np.linalg.norm(current_w_r)*np.linalg.norm(true_w))
            
            
            estimate_w_o[ite].append(m_o)
            estimate_w[ite].append(m)
            estimate_w_r[ite].append(m_r)
            
            print('Samples so far: ' + str(i))
            
            # run_algo :
            psi_set_id_o = algos.point_greedy(w_samples_o, b, data_psi_set)
            psi_set_id = algos.point_greedy(w_samples, b, data_psi_set)
            psi_set_id_r = algos.point_greedy(w_samples_r, b, data_psi_set)
            
            
            for j in range(b):
                
                o_psi, _, o_s = get_feedback(data_psi_set[psi_set_id_o[j]], true_w, true_w)
                psi, s, t_s = get_feedback(data_psi_set[psi_set_id[j]], target_w, true_w)
                psi_r, s_r, t_s_r = get_feedback(data_psi_set[psi_set_id_r[j]], target_w, true_w)
            
                oracle_psi_set.append(o_psi)
                oracle_s_set.append(o_s)
                
                psi_set.append(psi)
                s_set.append(s)
                o_s_set.append(t_s)
                
                
                psi_set_r.append(psi_r)
                s_set_r.append(s_r)  
                
                i+=1
                
            print(w_sampler_r.A.shape)
                
            aaa = np.array(w_sampler_r.A)
            bbb = np.array(w_sampler_r.y) 
            print(aaa.shape)
            print(bbb.shape)
                
        oracle_w_sampler.A = oracle_psi_set
        oracle_w_sampler.y = np.array(oracle_s_set).reshape(-1,1)
        w_samples_o = oracle_w_sampler.sample(M)
            
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1,1)
        w_samples = w_sampler.sample(M)
            
        w_sampler_r.A = psi_set_r
        w_sampler_r.y = np.array(s_set_r).reshape(-1,1)
        num_of_set = len(w_sampler_r.A)
        Outlierscore = np.zeros(num_of_set)
        
        
        w_samples_r = w_sampler_r.sample(M)
        
        
        
        mean_w_samples_o = np.mean(w_samples_o,axis=0)
        mean_w_samples = np.mean(w_samples,axis=0)
        mean_w_samples_r = np.mean(w_samples_r,axis=0)
        
        
        
        current_w_o = mean_w_samples_o/np.linalg.norm(mean_w_samples_o)
        current_w = mean_w_samples/np.linalg.norm(mean_w_samples)
        current_w_r = mean_w_samples_r/np.linalg.norm(mean_w_samples_r)
        
        m_o = np.dot(current_w_o, true_w)/(np.linalg.norm(current_w_o)*np.linalg.norm(true_w))
        m = np.dot(current_w, true_w)/(np.linalg.norm(current_w)*np.linalg.norm(true_w))
        
        
        ransac_w = RANSAC(0.2, np.array(psi_set), np.array(s_set_r).reshape(-1,1), 100, 10)
        ransac_m = np.dot(ransac_w, true_w)/(np.linalg.norm(ransac_w)*np.linalg.norm(true_w))
        
        
        print('*******stoc noisy ratio****************')
        print(len(np.where(np.array(o_s_set)!=np.array(s_set))[0])/100)
        print('***********************')
        
        print('oracle cosine similarity ' +str(m_o))
        print('greedy cosine similarity ' +str(m))
        print('ransac cosine similarity ' +str(ransac_m))
        
        

        

        
            
if __name__ == "__main__":
    RANSAC_PBL_Toy()