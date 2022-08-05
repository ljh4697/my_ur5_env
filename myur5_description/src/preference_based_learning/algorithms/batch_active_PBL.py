from algorithms.PBL_algorithm import PBL_model
from sampling import Sampler
import numpy as np
from simulation_utils import bench_run_algo


class batch_active_params_error(Exception):
    def __init__(self):
        super().__init__('it\'s not proper batch active params keys')



class batch_active_PBL(PBL_model):
    
    def __init__(self, simulation_object, batch_active_params):
        
        super().__init__(simulation_object)
        
        if list(batch_active_params.keys()) != ["method", "samples_num", "pre_greedy_nums"]:
            raise batch_active_params_error
        
        
        ''' hyper parameter ###############################################'''

        self.method = batch_active_params["method"]
        self.M = batch_active_params["samples_num"]
        self.B = batch_active_params["pre_greedy_nums"]
        
        '''################################################################'''
        

        
        self.w_sampler = Sampler(self.simulation_object.num_of_features)
        self.w_samples = self.w_sampler.sample(self.M)
        self.hat_theta_D = np.mean(self.w_samples,axis=0)
        

        
        
        
    def update_param(self, t):
        self.w_sampler.A = self.actions_s
        self.w_sampler.y = np.array(self.reward_s).reshape(-1,1)
        self.w_samples = self.w_sampler.sample(self.M)

        self.hat_theta_D = np.mean(self.w_samples,axis=0)
        
    def select_single_action(self, step):
        lower_input_bound = [x[0] for x in self.simulation_object.feed_bounds]
        upper_input_bound = [x[1] for x in self.simulation_object.feed_bounds]
        
        if step == 0:
            inputA = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(1, 2*self.simulation_object.feed_size))
            inputB = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(1, 2*self.simulation_object.feed_size))

            selected_actions = np.zeros((1, self.d))
        
            self.simulation_object.feed(inputA)
            phi_A = self.simulation_object.get_features()    
            
            self.simulation_object.feed(inputB)
            phi_B = self.simulation_object.get_features()
            
            selected_actions[0, :] = np.array(phi_A) - np.array(phi_B)
        else:
            inputA, inputB = bench_run_algo('nonbatch', self.simulation_object, self.w_samples, self.b, self.B)
            
            selected_actions = np.zeros((1, self.d))
        
            self.simulation_object.feed(inputA)
            phi_A = self.simulation_object.get_features()    
            
            self.simulation_object.feed(inputB)
            phi_B = self.simulation_object.get_features()
            
            selected_actions[0, :] = np.array(phi_A) - np.array(phi_B)
    
    
    
    def select_batch_actions(self, step, b):
        lower_input_bound = [x[0] for x in self.simulation_object.feed_bounds]
        upper_input_bound = [x[1] for x in self.simulation_object.feed_bounds]
        
        if step == 0:
            inputA_set = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(b, 2*self.simulation_object.feed_size))
            inputB_set = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(b, 2*self.simulation_object.feed_size))

            selected_actions = np.zeros((b, self.d))
            
            for i in range(b):
            
                self.simulation_object.feed(inputA_set[i])
                phi_A = self.simulation_object.get_features()    
                
                self.simulation_object.feed(inputB_set[i])
                phi_B = self.simulation_object.get_features()
                
                selected_actions[i, :] = np.array(phi_A) - np.array(phi_B)
        else:
            inputA_set, inputB_set = bench_run_algo(self.method, self.simulation_object, self.w_samples, b, self.B)
            
            selected_actions = np.zeros((b, self.d))
            
            for i in range(b):
            
                self.simulation_object.feed(inputA_set[i])
                phi_A = self.simulation_object.get_features()    
                
                self.simulation_object.feed(inputB_set[i])
                phi_B = self.simulation_object.get_features()
                
                selected_actions[i, :] = np.array(phi_A) - np.array(phi_B)
        
        return selected_actions
    
    
    