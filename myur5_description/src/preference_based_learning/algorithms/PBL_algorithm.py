import numpy as np


class PBL_model(object):
    def __init__(self, simulation_object):
        
        self.simulation_object = simulation_object
        self.d = simulation_object.num_of_features

        ''' predefined data#####################################################'''
        
        data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/ctrl_samples/' + self.simulation_object.name + '.npz')
        self.PSI = data['psi_set']
        self.inputs_set = data['inputs_set']
        features_data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/ctrl_samples/' + self.simulation_object.name + '_features'+'.npz')
        self.predefined_features = features_data['features']
        
        '''######################################################################'''
        
        self.action_s = []
        self.reward_s = []
    
    def update_param(self):
        raise NotImplementedError("must implement udate param method")
    def select_single_action(self):
        raise NotImplementedError("must implement select single action method")
    def select_batch_actions(self):
        raise NotImplementedError("must implement select single action method")
        
            
    def test(self):
        print("hello")
    
    
    
    
    
    
    
    