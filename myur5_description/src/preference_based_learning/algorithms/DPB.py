
from re import A
import numpy as np
from scipy.optimize import fmin_slsqp
import matplotlib.pyplot as plt 
from tqdm import trange
from algorithms.PBL_algorithm import PBL_model
import time


def mu(x, theta):
    return 1/(1+np.exp(-np.dot(x, theta)))


def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1
    else: return 0
    
    
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



class DPB_params_error(Exception):
    def __init__(self):
        super().__init__('it\'s not proper DPB params keys')
 
class DPB(PBL_model):
    def __init__(self, simulation_object, DPB_params, env='simulated'):
        super().__init__(simulation_object, env)
        if list(DPB_params.keys()).sort() != ["regularized_lambda", "c_mu", "k_mu",
                                       "discounting_factor", "param_U", "action_U",
                                       "delta", "reward_U", "exploration_weight"].sort():
            raise DPB_params_error
        
        ''' hyper parameter ###############################################'''
        
        
        self.regularized_lambda = DPB_params["regularized_lambda"]
        self.c_mu = DPB_params["c_mu"] ; self.k_mu = DPB_params["k_mu"]
        self.gamma = DPB_params["discounting_factor"]
        self.S = DPB_params["param_U"] 
        self.L = DPB_params["action_U"] # D = L
        self.delta = DPB_params["delta"]
        self.m = DPB_params["reward_U"]
        self.alpha = DPB_params["exploration_weight"] 
        
        '''################################################################'''
        
        
        self.D_rho = 0
        self.hat_theta_D = np.zeros(simulation_object.num_of_features)
        
        
        self.N_gamma = self.N_gamma_(self.gamma)
        #self.V_t = (self.regularized_lambda/self.c_mu)*np.identity(self.d)
        self.V_t = self.regularized_lambda*np.identity(self.d)
        


    def N_gamma_(self, gamma):
        
        return np.log(1/(1-gamma))/(1-gamma)
        
    
    def get_alpha(self, t):
        
        left_term = (self.S*(self.L)**2*self.gamma**(self.N_gamma))/(self.c_mu*np.sqrt(self.regularized_lambda)*(1-self.gamma))
        middle_term = (np.sqrt(self.regularized_lambda)*self.S)/self.c_mu
        right_term = (1/self.c_mu)*np.sqrt(2*np.log(1/self.delta)+self.d*np.log(1+((self.L**2)*(1-self.gamma**t))/(self.d*self.regularized_lambda*(1-self.gamma))))
        
        return left_term + middle_term + right_term
    



    def select_batch_actions(self, step, b):
        
        
        
        def Matrix_Norm(A, V):
            '''calculate ||A||_V'''
            AV = np.matmul(A, V)
            result = np.zeros(A.shape[0])
            
            for i in range(A.shape[0]):
                result[i] = np.sqrt(np.dot(AV[i,:],A[i,:]))
                
            return result
        

        
        given_actions = self.PSI
        z = self.simulation_object.feed_size
        
        if step == 0:
            random_initialize = np.random.randint(0, len(given_actions), b)
            
            selected_actions = given_actions[random_initialize]
            inputs_set = self.inputs_set[random_initialize]
            selected_ids = random_initialize
            
            for i in range(b):
                self.update_V_t(selected_actions[i])
            
        else:
            selected_actions = []
            inputs_set = []
            selected_ids = []
            D_rho = self.get_alpha(step)*self.alpha # alpha * exploration weight = final alpha (D_rho)
            
            empirical_reward  =np.maximum(np.dot(given_actions, self.hat_theta_D ),-np.dot(given_actions, self.hat_theta_D )) 
            INV_V_t = np.linalg.inv(self.V_t)
            
            # compute UCB score reward + exploration bonus
            XW_rho = empirical_reward + D_rho*Matrix_Norm(given_actions, INV_V_t)
            selected_ids = np.argsort(-XW_rho)[:b] # reverse 하여 top b action을 뽑기 위해 -부호 사용
 
            selected_actions = given_actions[selected_ids]
            inputs_set = self.inputs_set[selected_ids]
            
            for i in range(b):
                self.update_V_t(selected_actions[i])
            
            
            
        #  밴치마킹 환경이랑 avoid 환경이랑 inputs_set 표현 방식이 달라 조건문으로 나눠서 출력   
        if self.simulation_object.name == "avoid":
            return selected_actions, selected_ids, selected_ids
        else:
            return selected_actions, inputs_set[:, :z], inputs_set[:, z:]
        
        

    def update_V_t(self, A_t):
        A_t = A_t.reshape(self.d, -1)
        self.V_t = np.matmul(A_t, A_t.T) + self.gamma*self.V_t + self.regularized_lambda*(1-self.gamma)*np.identity(self.d)



    def update_param(self, t):
        if t == 0:
            return
        
        def regularized_negative_log_likelihood(theta):
            cross_entropy = -np.sum(np.array(self.gamma**np.arange(t,0,-1))*(np.array(self.reward_s)*np.log(mu(self.action_s, theta))
                                                        +(1-np.array(self.reward_s))*np.log(1-mu(self.action_s, theta))))
            regularized_term = (self.regularized_lambda/2)*np.linalg.norm(theta)**2
            
            return cross_entropy + regularized_term
    

        def ieq_const(theta):
            return self.S-np.linalg.norm(theta)
        
        
        self.hat_theta_D = fmin_slsqp(regularized_negative_log_likelihood, np.zeros(self.d),
                        ieqcons=[ieq_const],
                        iprint=0)

        self.hat_theta_D = self.hat_theta_D/np.linalg.norm(self.hat_theta_D)

