from sklearn import tree
from simulation_utils import create_env, perform_best
import sys
import numpy as np
from tqdm import trange
from run_optimizer import get_opt_f, get_abs_opt_f, get_opt_id

import rospy
import moveit_commander
import moveit_msgs.msg
import os; import sys


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from test_mesh_pickandplace import create_environment
import control_planning_scene


'''tosser'''

'''
task = 'tosser'

#true_w = [0.1,0.03725074,0.00664673,0.80602143] #(green)
 #true_w = [0.01, 0.2 ,0.1, 0.30602143] #(red)
true_w = [0.1,0.03725074,0.2664673,0.80602143] #(green with large flips)
true_w = true_w/np.linalg.norm(true_w)
'''

'''driver'''

#task = 'driver'




planning_scene_1 = control_planning_scene.control_planning_scene()

env, grasp_point, approach_direction, objects_co, neutral_position = create_environment(planning_scene_1)

display_trajectory_data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/sampled_trajectories/test_display_trajectory.npz', allow_pickle=True)
#start_trajectory_data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/sampled_trajectories/test_trajectory_start.npz', allow_pickle=True)


moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node("move_group_python_test", anonymous=True)

robot = moveit_commander.RobotCommander()
mv_group = moveit_commander.MoveGroupCommander("manipulator")

eef_link = mv_group.get_end_effector_link()
touch_links = robot.get_link_names(group="hand")





display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=3)

planning_scene_1.remove_object(objects_co['milk'])
planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)







'''avoiding'''

task = 'avoid'
#true_w = [0.7, 0.01, 0.7, 0.02] # 전진, 충돌 후 후진

#true_w = [0.7, 0.7, 0.7, 0.02] # 후진




#true_w = [0.15, -0.7, 0.7, 0.3] # 전진, 충돌 후 전진

#true_w = [0.7, 0.7, 0.8, 0.01] # 후진

#true_w = [0.7, -0.2, 0.7, -0.06] # 전진, 피하면서 전진

true_w = [1, 0, 0, 0]

#true_w = true_w/np.linalg.norm(true_w)


simulation_object = create_env(task)


features_data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/ctrl_samples/' + simulation_object.name + '_features'+'.npz', allow_pickle=True)
data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/ctrl_samples/' + simulation_object.name + '.npz', allow_pickle=True)
inputs_set = data['inputs_set']
start_trajectory_data = data['start_set']
            
display_trajectory_data= inputs_set
predefined_features = features_data['features']


true_opt_id = get_opt_id(predefined_features, true_w)


print(np.argmax(predefined_features[:,0]))
print(true_opt_id)

exit()


print(str(true_opt_id) + "@@@@@@@@@@@@@@@@@@@@@@@")
print(display_trajectory_data.shape)


display_trajectory = moveit_msgs.msg.DisplayTrajectory()
display_trajectory.model_id = 'ur5'
display_trajectory.trajectory.append(display_trajectory_data[true_opt_id][0])
display_trajectory.trajectory.append(display_trajectory_data[true_opt_id][1])
display_trajectory.trajectory.append(display_trajectory_data[true_opt_id][2])
display_trajectory.trajectory.append(display_trajectory_data[true_opt_id][3])

display_trajectory.trajectory_start=start_trajectory_data[true_opt_id][0]

attached_co = moveit_msgs.msg.AttachedCollisionObject()

attached_co.object = objects_co['milk']
attached_co.link_name = eef_link
attached_co.touch_links = touch_links
display_trajectory.trajectory_start.attached_collision_objects.append(attached_co)

for i in range(4):
    display_trajectory_publisher.publish(display_trajectory)
    rospy.sleep(0.1)