
from re import A
import numpy as np
from scipy.optimize import fmin_slsqp
import matplotlib.pyplot as plt 
from tqdm import trange
from algorithms.PBL_algorithm import PBL_model
import time


def mu(x, theta):
    return 1/(1+np.exp(-np.dot(x, theta)))


class DPB_params_error(Exception):
    def __init__(self):
        super().__init__('it\'s not proper DPB params keys')
 
class DPB2(PBL_model):
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
        self.V_t = self.regularized_lambda*np.identity(self.d)
        
        self.tilde_W_t = (self.regularized_lambda/self.c_mu)*np.identity(self.d)




    def N_gamma_(self, gamma):
        
        return np.log(1/(1-gamma))/(1-gamma)
        
    
    def get_alpha(self, t):
        
        left_term = (self.S*(self.L)**2*self.gamma**(self.N_gamma))/(self.c_mu*np.sqrt(self.regularized_lambda)*(1-self.gamma))
        middle_term = (np.sqrt(self.regularized_lambda)*self.S)/self.c_mu
        right_term = (1/self.c_mu)*np.sqrt(2*np.log(1/self.delta)+self.d*np.log(1+((self.L**2)*(1-self.gamma**t))/(self.d*self.regularized_lambda*(1-self.gamma))))
        
        return left_term + middle_term + right_term
        
    
    


    def select_batch_actions(self, step, b):
        
        
        
        def Matrix_Norm_ew(A, V):
            '''calculate ||A||_V'''
            AV = np.matmul(A, V)
            
            result = np.sqrt(np.sum(AV*A, axis=1))
                
            return result
        
        given_actions = self.PSI
        z = self.simulation_object.feed_size
        
        if step == 0:
            selected_actions = []
            inputs_set = []
            selected_ids = []
            
            
            '''update V_t ''' 
            for i in range(b):
                random_initialize = np.random.randint(0, len(given_actions), 1)[0]
            

                arm_selector = False; a = 0
                while not arm_selector:
                    if random_initialize not in selected_ids:
                        selected_ids.append(random_initialize)
                        arm_selector = True
                    else:
                        a += 1
                
                self.update_V_t(given_actions[random_initialize])

            selected_ids = np.array(selected_ids)
            selected_actions = given_actions[selected_ids]
            inputs_set = self.inputs_set[selected_ids]
            
            
        else:
            selected_actions = []
            inputs_set = []
            selected_ids = []
            D_rho = self.get_alpha(step)*self.alpha # alpha * exploration weight = final alpha (D_rho)
            
            empirical_reward  =np.maximum(np.dot(given_actions, self.hat_theta_D ),-np.dot(given_actions, self.hat_theta_D )) 
            
            '''update V_t'''
            for i in range(b):
                INV_V_t = np.linalg.inv(self.V_t)
                # compute UCB score reward + exploration bonus
                XW_rho = empirical_reward + D_rho*Matrix_Norm_ew(given_actions, INV_V_t)
                
                arg_XW_rho = np.argsort(-XW_rho)
                arm_selector = False; a = 0
                while not arm_selector:
                    if arg_XW_rho[a] not in selected_ids:
                        selected_ids.append(arg_XW_rho[a])
                        arm_selector = True
                    else:
                        a += 1
                    
                self.update_V_t(given_actions[arg_XW_rho[a]])

                
            selected_ids = np.array(selected_ids)
            selected_actions = given_actions[selected_ids]
            inputs_set = self.inputs_set[selected_ids]
            
            
    
        if self.simulation_object.name == "avoid":
            return selected_actions, selected_ids, selected_ids
        else:
            return selected_actions, inputs_set[:, :z], inputs_set[:, z:]
        
        
    def generate_unitball_actions(self):
            r = np.random.uniform(low=0, high=1, size=6)
            a = np.random.uniform(low=0, high=3.14*2, size=6)
            actions= np.array([r*np.cos(a), r*np.sin(a)]).T
            
            # actions shape = (6, 2)
            return actions
        
    def update_V_t(self, A_t):
        A_t = A_t.reshape(self.d, -1)
        self.V_t = np.matmul(A_t, A_t.T) + self.gamma*self.V_t + self.regularized_lambda*(1-self.gamma)*np.identity(self.d)

    def compute_tilde_w_t(self, A_t):
        
        self.tilde_W_t = np.matmul(A_t, A_t.T) + self.gamma**2*self.tilde_W_t + (self.regularized_lambda/self.c_mu)*(1-self.gamma)*np.identity(self.d)



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



 