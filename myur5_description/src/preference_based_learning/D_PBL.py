
from ast import Pass
from re import A
import numpy as np
from scipy.optimize import fmin_slsqp
import matplotlib.pyplot as plt 
from tqdm import trange

from bandit_base import GLUCB

def mu(x, theta):
    return 1/(1+np.exp(-np.dot(x, theta)))

def true_param(t):
    # theta^*_t
    true_param = np.zeros(2)
    if t<=100:
        true_param[0] = 1; true_param[1] = 0
    elif 101<=t<=200:
        true_param[0] = -1; true_param[1] = 0
    elif 201<=t<=300:
        true_param[0] = 0; true_param[1] = 1
    elif 301<=t:
       true_param[0] = 0; true_param[1] = -1
        
    return true_param   

class PBL(GLUCB):
    
    
    def __init__(self):
        super().__init__()

    
    def D_PBL(self, iter):
        epsilon = 0.2
        c_mu = 1/5 ; L_mu = 1/4
        d = 2; n_actions = 6
        gamma = 0.93
        S = 1
        D = 1
        delta = 0.9
        m =0.73
        
        
        
        theta_t = []
        
        cumulative_reward = 0
        
        # initialize V_t
        V_t = self.regularized_lambda*np.identity(d)
        
        # initialize \hat_theta
        hat_theta_D = np.zeros(2)
        
        X_s = np.zeros(2)
        A_t = np.zeros(2)
        A_s = []
        reward_s = []
        g_reward_s = []
        actions_s = []
        g_actions_s = []
        parameter_archive = []
        g_parameter_archive = []
        

        
        
        # regret archive
        regret_D_PBL = [0]
        regret_random = [0]
        regret_greedy = [0]
        
        
        
        def regularized_log_likelihood(theta):
            return -(np.sum(np.array(gamma**np.arange(t,0,-1))*(np.array(reward_s)*np.log(mu(actions_s, theta))
                                                        +(1-np.array(reward_s))*np.log(1-mu(actions_s, theta))))-(self.regularized_lambda/2)*np.linalg.norm(theta)**2)
        
        def greedy_regularized_log_likelihood(theta):
            return -(np.sum((np.array(g_reward_s)*np.log(mu(g_actions_s, theta))
                                                        +(1-np.array(g_reward_s))*np.log(1-mu(g_actions_s, theta))))-(self.regularized_lambda/2)*np.linalg.norm(theta)**2)
        
        def g_t_theta(theta):
            
            T = len(actions_s)
            left_g = np.zeros_like(theta)
            for s in range(T):
                left_g += (gamma**(T-s))*mu(actions_s[s], theta)*actions_s[s]
                
            return (left_g + self.regularized_lambda*theta).reshape(-1, 1)
        

                
        def compute_alpha_T(delta):
            left_alpha = (1/c_mu)*np.sqrt((2*np.log(1/delta))+
                                    (d*np.log(1+((D**2)/(d*self.regularized_lambda))*((1-gamma**(t-1))/(1-gamma)))))
            
            right_alpha = ((L_mu*d)/c_mu)*np.sqrt(2*D**2*S**2*((1-gamma**(t-1))/(1-gamma))+
                                                  2*self.regularized_lambda*S**2)+np.sqrt(self.regularized_lambda)*S/c_mu
            
            
            return left_alpha + right_alpha
            
        
        def random_select(d):
            return np.random.randint(0,d,1)[0]
            
        def compute_regret(action):
            return np.max(mu(actions,param_star))-mu(action,param_star)
            
        
        for t in trange(iter):
            param_star = true_param(t)
            
            
            if not tuple(param_star) in theta_t:
                theta_t.append(tuple(param_star))
            actions = self.generate_unitball_actions()
            
            if t == 0:
                A_t = actions[np.random.randint(0,6,1)[0]]
                random_A_t = actions[np.random.randint(0,6,1)[0]]
                greedy_A_t = actions[np.random.randint(0,6,1)[0]]
                

            # compute theta_D
            else:
                    
                hat_theta_D = fmin_slsqp(regularized_log_likelihood, np.array([0, 0]), iprint=0)
                tilde_theta_D = hat_theta_D
                

                alpha_T = compute_alpha_T(delta)/40
                print(alpha_T)
                print('eee')
                # D_rho 값이 너무 크게 계산됨. D_rho = 0.1 정도로 나와야 적당한 것 같음
                
                # D_PBL select action
                
                A_t = actions[np.argmax(np.dot(actions, tilde_theta_D) #)]
                                       +alpha_T*np.sqrt(np.diag(np.matmul(np.matmul(actions, V_t), actions.T))))]
                
                #print(np.sqrt(np.diag(np.matmul(np.matmul(actions, V_t), actions.T))))
                #print(mu(actions,tilde_theta_D))
                
                
                ###################################################################################
                ######################### bench marking algorithms ################################
                ###################################################################################
                
                #######################
                # ranom ###############
                #######################
                random_A_t = actions[random_select(d)]
                
                #######################
                # epsilon greedy ######
                #######################
                
                if np.random.uniform(0,1) < epsilon:
                    g_hat_theta_D = actions[random_select(d)]
                else:
                    g_hat_theta_D = fmin_slsqp(greedy_regularized_log_likelihood, np.array([0, 0]), iprint=0)
                
                greedy_A_t = actions[np.argmax(mu(actions,g_hat_theta_D))]
                
                
                
                
                
            regret_D_PBL.append(regret_D_PBL[-1] + compute_regret(A_t))
            regret_random.append(regret_random[-1] + compute_regret(random_A_t))
            regret_greedy.append(regret_greedy[-1] + compute_regret(greedy_A_t))
            
            
            reward =  mu(A_t, param_star)
            reward_s.append(reward)
            
            g_reward = mu(greedy_A_t, param_star)
            g_reward_s.append(g_reward)
            
            cumulative_reward += reward
            
            actions_s.append(A_t)
            g_actions_s.append(greedy_A_t)
            
                
                
            if (t+1)%100==0:
                parameter_archive.append(tilde_theta_D)
                g_parameter_archive.append(g_hat_theta_D)
                
                
            A_t = A_t.reshape(-1, 1)
            V_t = np.matmul(A_t, A_t.T) + gamma*V_t + (1-gamma)*np.identity(d)
            
        parameter_archive = np.array(parameter_archive)
        g_parameter_archive = np.array(g_parameter_archive)
        
        theta_t = np.array(theta_t)
        
        
        
        
        
        
        
        
        
        # plot graph-
        figure, axes = plt.subplots(1,2, figsize=(10,5))
        draw_circle = plt.Circle((0, 0), 1, fill=False, color='red', zorder=0)
        axes[0].add_artist(draw_circle)
        axes[0].set_title('Circle')
        axes[0].set_xlim([-1.15, 1.15])
        axes[0].set_ylim([-1.15, 1.15])

        for i in range(len(theta_t)):
            if i == 0:
                axes[0].scatter(theta_t[i][0], theta_t[i][1], marker='v', zorder=1, color='blue', label='true')
            else:
                axes[0].scatter(theta_t[i][0], theta_t[i][1], marker='v', zorder=1, color='blue')
                
            axes[0].annotate(str(i+1), xy=(theta_t[i][0],theta_t[i][1]),xytext=(10, 10), textcoords='offset pixels')

        axes[0].plot(parameter_archive[:,0], parameter_archive[:,1], marker='D', zorder=2, color='red', linestyle='dashed', label='D_PBL')
        axes[0].plot(g_parameter_archive[:,0], g_parameter_archive[:,1], marker='o', zorder=2, color='orange', linestyle='dashed', label='greedy')
        axes[0].legend()        
        axes[0].set_aspect(1)
        
        axes[1].plot(np.arange(len(regret_D_PBL)), regret_D_PBL, color='red', label='D_PBL')
        axes[1].plot(np.arange(len(regret_D_PBL)), regret_greedy, color='orange', label='greedy')
        axes[1].plot(np.arange(len(regret_D_PBL)), regret_random, color='purple', label='random')
        axes[1].legend()        
        plt.show()

            
        return cumulative_reward
          
    
