from moveit_commander.robot import RobotCommander
import rospy
import control_planning_scene
import ur5e_plan
import ur5e_get_fk
import numpy as np
from sensor_msgs.msg import JointState

from test_mesh_pickandplace import create_environment
import os









class featuremapping(object):
    def __init__(self, planning_scene_1):
        super(featuremapping, self).__init__()
        #rospy.init_node("test_fk", anonymous=False)
        self.rc = RobotCommander()
        self.planning_scene_1 = planning_scene_1
        self.planning_ur5e = ur5e_plan.ur5e_plan()
        self.getfk = ur5e_get_fk.GetFK('ee_link', 'world')
        self.ur5e_js = self.planning_scene_1.get_current_joint_state()

    def compute_feature_map(self, step):
        feature_mapping = np.zeros(4)
        TABLE_HEIGHT = 0.8
        eef_height = 0
        t_distance_to_laptop = 0
        moving_distance = 0
        t_distance_to_user = 0 
        previous_position = np.zeros(3)
        
        for p in step:
            
            
            joint_position = np.zeros(12)
            joint_position[:6] = p
            self.planning_scene_1.get_planning_scene.robot_state.joint_state.position = joint_position
            eef_position = self.getfk.get_fk(self.ur5e_js)
            
            distance_to_laptop = np.linalg.norm(eef_position-self.object_position)
            distance_to_user = np.linalg.norm(eef_position-self.user_position)
            
            t_distance_to_laptop += distance_to_laptop
            t_distance_to_user += distance_to_user
            eef_height += eef_position[2] - TABLE_HEIGHT
            if np.sum(previous_position) != 0:
                moving_distance += np.linalg.norm(eef_position - previous_position) 
                    
            previous_position = eef_position
            
            
        eef_height/=len(step)
        t_distance_to_laptop/=len(step)
        t_distance_to_user/=len(step)
        feature_mapping[0] = eef_height
        feature_mapping[1] = t_distance_to_laptop
        feature_mapping[2] = moving_distance
        feature_mapping[3] = t_distance_to_user
        
        return feature_mapping
        
    def get_feature(self, objects_co, planning_trajectory = None):
        '''
        1. end_effector's height
        2. distance between eef and laptop
        3. moving distance
        4. distance between eef and user
        '''


        t_feature_mapping = np.zeros(4)        
        

        #env, grasp_point, approach_direction, objects_co, neutral_position = create_environment(self.planning_scene_1)
        
        
        
        self.object_position = np.zeros(3)
        self.object_position[0] = objects_co['laptop'].mesh_poses[0].position.x
        self.object_position[1] = objects_co['laptop'].mesh_poses[0].position.y
        self.object_position[2] = objects_co['laptop'].mesh_poses[0].position.z
        
        self.user_position = np.zeros(3)
        self.user_position[0] = objects_co['visualhuman'].mesh_poses[0].position.x
        self.user_position[1] = objects_co['visualhuman'].mesh_poses[0].position.y
        self.user_position[2] = objects_co['visualhuman'].mesh_poses[0].position.z + 0.9

        
        if type(planning_trajectory) !=  list and type(planning_trajectory) != np.ndarray:
            if planning_trajectory == None:
                planning_trajectory = np.load('./sampled_trajectories/pick_trajectories.npz', allow_pickle=True)['plan']
        #print(pick_trajectories.files[0])
        for step in planning_trajectory:

            if type(step) == list:
                # trajectory의 feature map (mean value)
                feature_mapping = self.compute_feature_map(step)
                t_feature_mapping += feature_mapping
                print(feature_mapping)
                
            if type(step) == str:
                print(step)
        t_feature_mapping 
                
        return  t_feature_mapping
    
    
def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    traj_path = os.path.join(dir_path,  'sampled_trajectories/planning_trajectory.npz')
    
    # choose one sample trajectory
    planning_trajectory = np.load(traj_path, allow_pickle=True)['plan'][0]
    
    print(len(planning_trajectory))
    get_feature_map = featuremapping()
    
    
    print('end_effectors height' + 'distance between eef and laptop' + 'moving distance' + 'distance between eef and user')
    features_sum = get_feature_map.get_feature(planning_trajectory=planning_trajectory)   
    
    # 각 step 에서 feature map values 들의 sum
    print('sum of featuremap')
    print(features_sum)
    
    


if __name__ == "__main__":
    main()