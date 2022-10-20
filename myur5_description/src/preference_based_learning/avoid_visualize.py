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




def visualization(opt_id, algo):

    task = 'avoid'
    objects_co = algo.objects_co
    display_trajectory_data = algo.inputs_set
    start_trajectory_data = algo.start_data

    
    #planning_scene_1 = control_planning_scene.control_planning_scene()

    #env, grasp_point, approach_direction, objects_co, neutral_position = create_environment(planning_scene_1)

    #display_trajectory_data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/sampled_trajectories/test_display_trajectory.npz', allow_pickle=True)
    #start_trajectory_data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/sampled_trajectories/test_trajectory_start.npz', allow_pickle=True)


    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("move_group_python_test", anonymous=True)

    robot = moveit_commander.RobotCommander()
    mv_group = moveit_commander.MoveGroupCommander("manipulator")

    eef_link = mv_group.get_end_effector_link()
    touch_links = robot.get_link_names(group="hand")

    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=3)

    #planning_scene_1.remove_object(objects_co['milk'])
    #planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)





    simulation_object = create_env(task)


    features_data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/ctrl_samples/' + simulation_object.name + '_features'+'.npz', allow_pickle=True)
    data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/ctrl_samples/' + simulation_object.name + '.npz', allow_pickle=True)
    inputs_set = data['inputs_set']
    start_trajectory_data = data['start_set']
                
    display_trajectory_data= inputs_set
    predefined_features = features_data['features']



    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.model_id = 'ur5'
    display_trajectory.trajectory.append(display_trajectory_data[opt_id][0])
    display_trajectory.trajectory.append(display_trajectory_data[opt_id][1])
    display_trajectory.trajectory.append(display_trajectory_data[opt_id][2])
    display_trajectory.trajectory.append(display_trajectory_data[opt_id][3])

    display_trajectory.trajectory_start=start_trajectory_data[opt_id][0]

    attached_co = moveit_msgs.msg.AttachedCollisionObject()

    attached_co.object = objects_co['milk']
    attached_co.link_name = eef_link
    attached_co.touch_links = touch_links
    display_trajectory.trajectory_start.attached_collision_objects.append(attached_co)

    for i in range(4):
        display_trajectory_publisher.publish(display_trajectory)
        rospy.sleep(0.1)