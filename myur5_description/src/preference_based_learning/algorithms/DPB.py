
from re import A
import numpy as np
from scipy.optimize import fmin_slsqp
import matplotlib.pyplot as plt 
from tqdm import trange
from algorithms.PBL_algorithm import PBL_model


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
        self.L = DPB_params["action_U"]
        self.delta = DPB_params["delta"]
        self.m = DPB_params["reward_U"]
        self.alpha = DPB_params["exploration_weight"] 
        
        '''################################################################'''
        
        
        self.D_rho = 0
        self.hat_theta_D = np.zeros(simulation_object.num_of_features)
        
        self.W_t = (self.regularized_lambda/self.c_mu)*np.identity(self.d)
        self.tilde_W_t = (self.regularized_lambda/self.c_mu)*np.identity(self.d)



    def D_c_delta(self, t):
        return (self.m/2)*np.sqrt(2*np.log(1/self.delta)+
                            self.d*np.log(1+((self.c_mu*(self.L**2)*(1-self.gamma**(2*t)))/(self.d*self.regularized_lambda*(1-self.gamma**2)))))

    def D_rho_delta(self, t):
        D_c = self.D_c_delta(t)
        return (2*self.k_mu/self.c_mu)*(D_c + np.sqrt(self.c_mu*self.regularized_lambda)*self.S+ 2*self.L**2*self.S*self.k_mu*np.sqrt(self.c_mu/self.regularized_lambda))



        
    def select_single_action(self, step):
            
            
        given_actions = self.PSI
        
        if step == 0:
            selected_actions = given_actions[np.random.randint(0, len(given_actions), 10)]
            
        else:
            self.D_rho = self.D_rho_delta(step)*self.alpha
            
            ucb_scores = np.maximum(np.dot(given_actions, self.hat_theta_D ),-np.dot(given_actions, self.hat_theta_D )) + self.D_rho*np.sqrt(np.diag(np.matmul(np.matmul(given_actions, self.W_t), given_actions.T)))
            selected_actions = given_actions[np.argmax(ucb_scores)]
        
        return selected_actions


    def select_batch_actions(self, step, b):
        
        given_actions = self.PSI
        z = self.simulation_object.feed_size
        
        if step == 0:
            random_initialize = np.random.randint(0, len(given_actions), 10)
            
            selected_actions = given_actions[random_initialize]
            inputs_set = self.inputs_set[random_initialize]
            selected_ids = random_initialize
            
            for i in range(b):
                self.compute_w_t(selected_actions[i])
            
        else:
            selected_actions = []
            inputs_set = []
            D_rho = self.D_rho_delta(step)*self.alpha
            
            empirical_reward  =np.maximum(np.dot(given_actions, self.hat_theta_D ),-np.dot(given_actions, self.hat_theta_D )) 
            #empirical_reward  = -np.dot(given_actions, self.hat_theta_D)
            
            
            # for i in range(b):
            #     XW_rho = empirical_reward + self.D_rho*np.sqrt(np.diag(np.matmul(np.matmul(given_actions, self.W_t), given_actions.T)))
            #     argmax_action = given_actions[np.argmax(XW_rho)]
            #     inputs_set = self.inputs_set[np.argsort(XW_rho)]
                
            #     selected_actions.append(argmax_action)
            #     self.compute_w_t(argmax_action)
            
            # selected_actions = np.array(selected_actions)
            
            XW_rho = empirical_reward + D_rho*np.sqrt(np.diag(np.matmul(np.matmul(given_actions, self.W_t), given_actions.T)))
            
            selected_ids = np.argsort(-XW_rho)[:b]
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
        
    def compute_w_t(self, A_t):
        
        self.W_t = np.matmul(A_t, A_t.T) + self.gamma*self.W_t + self.regularized_lambda*(1-self.gamma)*np.identity(self.d)

    def compute_tilde_w_t(self, A_t):
        
        self.tilde_W_t = np.matmul(A_t, A_t.T) + self.gamma**2*self.tilde_W_t + (self.regularized_lambda/self.c_mu)*(1-self.gamma)*np.identity(self.d)



    def update_param(self, t):
        if t == 0:
            return
        
        def regularized_log_likelihood(theta):
            
            return -(np.sum(np.array(self.gamma**np.arange(t,0,-1))*(np.array(self.reward_s)*np.log(mu(self.action_s, theta))
                                                        +(1-np.array(self.reward_s))*np.log(1-mu(self.action_s, theta))))-(self.regularized_lambda/2)*np.linalg.norm(theta)**2)
    

        def ieq_const(theta):
            return self.S-np.linalg.norm(theta)
        
        
        self.hat_theta_D = fmin_slsqp(regularized_log_likelihood, np.zeros(self.d),
                        ieqcons=[ieq_const],
                        iprint=0)





        
    def D_GLUCB(self, iter):
        # algorithm demo
        epsilon = 0.2
        c_mu = 1/5 ; k_mu = 1/4
        d = 2; 
        gamma = 0.93
        S = 1
        L = 1.4
        delta = 0.7
        m = 1
        
        
        
        theta_t = []
        
        cumulative_reward = 0
        
        # initialize W_t
        W_t = (self.regularized_lambda/c_mu)*np.identity(d)
        
        # initialize \hat_theta
        hat_theta_D = np.zeros(2)
        
        X_s = np.zeros(2)
        A_t = np.zeros(2)
        A_s = []
        reward_s = []
        g_reward_s = []
        action_s = []
        g_actions_s = []
        parameter_archive = []
        g_parameter_archive = []
        

        
        
        # regret archive
        regret_D_GLUCB = [0]
        regret_random = [0]
        regret_greedy = [0]
        
        
        
        def regularized_log_likelihood(theta):
            return -(np.sum(np.array(gamma**np.arange(t,0,-1))*(np.array(reward_s)*np.log(mu(action_s, theta))
                                                        +(1-np.array(reward_s))*np.log(1-mu(action_s, theta))))-(self.regularized_lambda/2)*np.linalg.norm(theta)**2)
        
        def greedy_regularized_log_likelihood(theta):
            return -(np.sum((np.array(g_reward_s)*np.log(mu(g_actions_s, theta))
                                                        +(1-np.array(g_reward_s))*np.log(1-mu(g_actions_s, theta))))-(self.regularized_lambda/2)*np.linalg.norm(theta)**2)
        
        def g_t_theta(theta):
            
            T = len(action_s)
            left_g = np.zeros_like(theta)
            for s in range(T):
                left_g += (gamma**(T-s))*mu(action_s[s], theta)*action_s[s]
                
            return (left_g + self.regularized_lambda*theta).reshape(-1, 1)
        
        def get_tilde_theta(theta):
            
            X = g_t_theta(hat_theta_D) - g_t_theta(theta)
            return np.linalg.norm(np.matmul(np.matmul(X.T, W_t), X))
        
        def ieq_const(theta):
            return S-np.linalg.norm(theta)
        
        def D_c_delta(delta):
            return (m/2)*np.sqrt(2*np.log(1/delta)+
                                 d*np.log(1+((c_mu*(L**2)*(1-gamma**(2*t)))/(d*self.regularized_lambda*(1-gamma**2)))))
        
        def D_rho_delta(delta):
            D_c = D_c_delta(delta)
            return (2*k_mu/c_mu)*(D_c + np.sqrt(c_mu*self.regularized_lambda)*S+ 2*L**2*S*k_mu*np.sqrt(c_mu/self.regularized_lambda))
        
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
                
                
                if np.linalg.norm(hat_theta_D) <= S:
                    tilde_theta_D = hat_theta_D
                else:
                    
                    print('--------------------')
                        
                    tilde_theta_D = fmin_slsqp(get_tilde_theta, np.array([0, 0]),
                                               ieqcons=[ieq_const], iprint=0) 
                
                
                D_rho = D_rho_delta(delta)/30
                
                # D_rho 값이 너무 크게 계산됨. D_rho = 0.1 정도로 나와야 적당한 것 같음
                
                # D_GLUCB select action
                A_t = actions[np.argmax(mu(actions,tilde_theta_D) #)]
                                       +D_rho*np.sqrt(np.diag(np.matmul(np.matmul(actions, self.W_t), actions.T))))]
                
                
                print(A_t)

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
                
                
                
                
                
            regret_D_GLUCB.append(regret_D_GLUCB[-1] + compute_regret(A_t))
            regret_random.append(regret_random[-1] + compute_regret(random_A_t))
            regret_greedy.append(regret_greedy[-1] + compute_regret(greedy_A_t))
            
            
            reward =  mu(A_t, param_star)
            reward_s.append(reward)
            
            g_reward = mu(greedy_A_t, param_star)
            g_reward_s.append(g_reward)
            
            cumulative_reward += reward
            
            action_s.append(A_t)
            g_actions_s.append(greedy_A_t)
            
                
            if (t+1)%100==0:
                parameter_archive.append(tilde_theta_D)
                g_parameter_archive.append(g_hat_theta_D)
                
            A_t = A_t.reshape(-1, 1)
            self.compute_w_t(A_t)
            
            
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

        axes[0].plot(parameter_archive[:,0], parameter_archive[:,1], marker='D', zorder=2, color='red', linestyle='dashed', label='D_GLUCB')
        axes[0].plot(g_parameter_archive[:,0], g_parameter_archive[:,1], marker='o', zorder=2, color='orange', linestyle='dashed', label='greedy')
        axes[0].legend()        
        axes[0].set_aspect(1)
        
        axes[1].plot(np.arange(len(regret_D_GLUCB)), regret_D_GLUCB, color='red', label='D_GLUCB')
        axes[1].plot(np.arange(len(regret_D_GLUCB)), regret_greedy, color='orange', label='greedy')
        axes[1].plot(np.arange(len(regret_D_GLUCB)), regret_random, color='purple', label='random')
        axes[1].legend()        
        plt.show()

            
        return cumulative_reward
            
            



