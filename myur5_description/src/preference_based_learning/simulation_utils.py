import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import trajectory_msgs.msg
from test_mesh_pickandplace import create_environment
import control_planning_scene
import scipy.optimize as opt
import algos



def compute_entropy(p):
    return np.sum(-p*np.log(p))/2
    
    







## automatically update by calculating distance from true_w
def predict_feedback(psi_set, current_w):
    feedback_entropy_set = []
    s_set = []
    for psi in psi_set:
        s = 0
        feedback_probability = np.zeros(2)
        feedback_probability[0] = 1/(1+np.exp(-1*np.dot(psi, current_w)))
        feedback_probability[1] = 1 - feedback_probability[0]
        while s==0:
            
            
            if np.dot(psi, current_w)>0:
                s =1
            else:
                s =-1
        feedback_entropy_set.append(compute_entropy(feedback_probability))
        s_set.append(s)
    return feedback_entropy_set, s_set




def get_feedback(psi, true_w):
     
    s = 0
    while s==0:
        
        
        if np.dot(psi, true_w)>0:
            s =1
        else:
            s =-1

    return psi, s

# update by user's online feedback
def get_user_feedback(psi, idx, objects_co):
    



    display_trajectory_data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/sampled_trajectories/display_trajectory.npz', allow_pickle=True)
    start_trajectory_data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/sampled_trajectories/trajectory_start.npz', allow_pickle=True)


    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("move_group_python_test", anonymous=True)

    robot = moveit_commander.RobotCommander()
    mv_group = moveit_commander.MoveGroupCommander("manipulator")

    eef_link = mv_group.get_end_effector_link()
    touch_links = robot.get_link_names(group="hand")




    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=3)



    s = 0


    while s==0:


        selection = input('A/B to watch, 1/2 to vote: ').lower()

        if selection == 'a':

            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.model_id = 'ur5'
            display_trajectory.trajectory.append(display_trajectory_data['plan'][idx*2][0])
            display_trajectory.trajectory.append(display_trajectory_data['plan'][idx*2][1])
            display_trajectory.trajectory.append(display_trajectory_data['plan'][idx*2][2])
            display_trajectory.trajectory.append(display_trajectory_data['plan'][idx*2][3])

            display_trajectory.trajectory_start=start_trajectory_data['plan'][idx*2][0]

            attached_co = moveit_msgs.msg.AttachedCollisionObject()

            attached_co.object = objects_co['milk']
            attached_co.link_name = eef_link
            attached_co.touch_links = touch_links
            display_trajectory.trajectory_start.attached_collision_objects.append(attached_co)

            for i in range(4):
                display_trajectory_publisher.publish(display_trajectory)
                rospy.sleep(0.1)

        elif selection == 'b':

            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.model_id = 'ur5'
            display_trajectory.trajectory.append(display_trajectory_data['plan'][idx*2+1][0])
            display_trajectory.trajectory.append(display_trajectory_data['plan'][idx*2+1][1])
            display_trajectory.trajectory.append(display_trajectory_data['plan'][idx*2+1][2])
            display_trajectory.trajectory.append(display_trajectory_data['plan'][idx*2+1][3])

            display_trajectory.trajectory_start=start_trajectory_data['plan'][idx*2+1][0]

            attached_co = moveit_msgs.msg.AttachedCollisionObject()

            attached_co.object = objects_co['milk']
            attached_co.link_name = eef_link
            attached_co.touch_links = touch_links
            display_trajectory.trajectory_start.attached_collision_objects.append(attached_co)

            for i in range(4):
                display_trajectory_publisher.publish(display_trajectory)
                rospy.sleep(0.1)

        elif selection == '1':
            s = 1
        elif selection == '2':
            s = -1

    return psi, s








def run_algo(method, w_samples, b=10, B=200):
    if method == 'nonbatch':
        return algos.nonbatch(w_samples)
    if method == 'greedy':
        return algos.greedy(w_samples, b)
    elif method == 'medoids':
        return algos.medoids(w_samples, b, B)
    elif method == 'boundary_medoids':
        return algos.boundary_medoids(w_samples, b, B)
    elif method == 'successive_elimination':
        return algos.successive_elimination(w_samples, b, B)
    elif method == 'random':
        return algos.random(w_samples)
    else:
        print('There is no method called ' + method)
        exit(0)


def func(ctrl_array, *args):
    simulation_object = args[0]
    w = np.array(args[1])
    simulation_object.set_ctrl(ctrl_array)
    features = simulation_object.get_features()
    return -np.mean(np.array(features).dot(w))

def perform_best(simulation_object, w, iter_count=10):
    u = simulation_object.ctrl_size
    lower_ctrl_bound = [x[0] for x in simulation_object.ctrl_bounds]
    upper_ctrl_bound = [x[1] for x in simulation_object.ctrl_bounds]
    opt_val = np.inf
    for _ in range(iter_count):
        temp_res = opt.fmin_l_bfgs_b(func, x0=np.random.uniform(low=lower_ctrl_bound, high=upper_ctrl_bound, size=(u)),
                                    args=(simulation_object, w), bounds=simulation_object.ctrl_bounds, approx_grad=True)
        print(temp_res[1])
        if temp_res[1] < opt_val:
            optimal_ctrl = temp_res[0]
            opt_val = temp_res[1]
    simulation_object.set_ctrl(optimal_ctrl)
    keep_playing = 'y'
    while keep_playing == 'y':
        keep_playing = 'u'
        simulation_object.watch(1)
        while keep_playing != 'n' and keep_playing != 'y':
            keep_playing = input('Again? [y/n]: ').lower()
    return -opt_val
