from copyreg import pickle
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
import control_planning_scene
from test_mesh_pickandplace import create_environment





class PBL_model(object):
    def __init__(self, simulation_object, env='simulated'):
        
        self.simulation_object = simulation_object
        self.d = simulation_object.num_of_features

        ''' predefined data#####################################################'''
        
        data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/ctrl_samples/' + self.simulation_object.name + '.npz', allow_pickle=True)
        self.PSI = data['psi_set']
        self.inputs_set = data['inputs_set']
        features_data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/ctrl_samples/' + self.simulation_object.name + '_features'+'.npz', allow_pickle=True)
        self.predefined_features = features_data['features']
        
        '''######################################################################'''
        
        self.action_s = []
        self.reward_s = []
        
        
        if simulation_object.name == 'avoid' and env =='real':
            # make env from mujoco world
            
            planning_scene_1 = control_planning_scene.control_planning_scene()

            env, grasp_point, approach_direction, self.objects_co, neutral_position = create_environment(planning_scene_1)

            planning_scene_1.remove_object(self.objects_co['milk'])
            planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
            self.start_data = data['start_set']
            
            

            
            
    
    def update_param(self):
        raise NotImplementedError("must implement udate param method")
    def select_single_action(self):
        raise NotImplementedError("must implement select single action method")
    def select_batch_actions(self):
        raise NotImplementedError("must implement select single action method")
        
            
    def test(self):
        print("hello")
    
    
    
    
    
    
    
    