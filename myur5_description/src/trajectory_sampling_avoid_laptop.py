#!/usr/bin/env python

import numpy as np
import trimesh
import os
import copy

from moveit_commander import (
    RobotCommander,
    PlanningSceneInterface,
    roscpp_initialize,
    roscpp_shutdown,
)
from moveit_commander import move_group
from moveit_commander.robot import RobotCommander
import rospy
import moveit_msgs.srv
import moveit_msgs.msg
import geometry_msgs.msg
from moveit_msgs.msg import Grasp
from math import pi, tau
import moveit_commander
import sys
from moveit_msgs.msg import AttachedCollisionObject
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import PlaceLocation
from tf.transformations import quaternion_from_euler
import numpy as np
import robosuite as suite

from std_msgs.msg import Header
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState

from robosuite import load_controller_config
from tf.transformations import quaternion_from_euler


import control_planning_scene
import ur5e_plan
from ur5e_dynamic_world import dynamic_world
from test_mesh_pickandplace import create_environment
from test_mesh_pickandplace import copy_pose
from test_mesh_pickandplace import set_approach_position
from test_mesh_pickandplace import direction_distance
from test_mesh_pickandplace import cross_product
from test_mesh_pickandplace import revolute_degree

import argparse
from get_feature import featuremapping

def main():
    pick_trajectory = list()
    planning_trajectory = [[] for i in range(args['num_trajectories'])]
    
    rospy.init_node("tutorial_ur5e", anonymous=True)
    robot = RobotCommander()
    # arm_move_group = moveit_commander.MoveGroupCommander("ur5e_arm")
    # arm_move_group.set_start_state_to_current_state()
    
    planning_scene_1 = control_planning_scene.control_planning_scene()
    get_feature_map = featuremapping(planning_scene_1)
    
    planning_ur5e = ur5e_plan.ur5e_plan()

    #env, box_position, obejct_01, neutral_position = 
    env, grasp_point, approach_direction, objects_co, neutral_position = create_environment(planning_scene_1)
    #approach_direction['milk'] = np.zeros(3)
    #approach_direction['milk'][0] = 1
    approach_position ={}
    approach_position['milk'] = grasp_point['milk'].copy()
    approach_position['milk'][0]-=0.25
    place_position = grasp_point['milk'].copy()
    
    # x 0.03~0.17 y = 0.35~0.7 , z = 0.03~0.75 suitable distance
    # define mid point (point to pass)
    
    place_position[0] += 0.05 ; place_position[1] += 1.0 ; place_position[2] += 0.1
    
    revolute_degree(approach_direction['milk'])



    #start planning
    r_last_position, pose_goal, plan1 = planning_ur5e.pose_plan_path(object_pose=approach_position['milk'],approach_direction="horizon")
    

    pick_trajectory.append(plan1)
    #input("press \"enter\" to open gripper")

    planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
    planning_scene_1.r_open_gripper()
    planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
    
    plan2 = "open"
    pick_trajectory.append(plan2)

    #input("press \"enter\" to cartesian path")


    pose_goal.position.x += 0.1
    

    r_last_position, plan3=planning_ur5e.plan_cartesian_path(wpose=pose_goal)

    pick_trajectory.append(plan3)
    
    #input("press \"enter\" to grasp and attach box to gripper")
    
    planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
    planning_scene_1.r_close_gripper()
    planning_scene_1.attach_object(objects_co['milk'])
    planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
    
    plan4 = "close"
    pick_trajectory.append(plan4)
    

    #input("press \"enter\" to retreat")

    planning_scene_1.r_open_gripper()
    planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
    pose_goal.position.z += 0.1
    
    pick_last_position, plan5=planning_ur5e.plan_cartesian_path(wpose=pose_goal)
    
    pick_trajectory.append(plan5)
    
    pick_trajectory = np.array(pick_trajectory , dtype=object)
    
    # save pick trajectories 
    if not os.path.isdir('sampled_trajectories'):
        os.mkdir('sampled_trajectories')
    
    np.savez("./sampled_trajectories/pick_trajectories.npz" , plan=pick_trajectory)
    print("complete to save pick trajectories")
    
    
    
    input("press \"enter\" to approach midpoint position")

    
    for i in range(args['num_trajectories']):
        input("sampling trajectory")
        
        planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=pick_last_position)
        planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
        midpoint = grasp_point['milk'].copy()
        midpoint[0] += np.random.uniform(0.03, 0.17) ; midpoint[1] += np.random.uniform(0.34, 0.70) ; midpoint[2] += np.random.uniform(0.03, 0.75)
        
        r_last_position, pose_goal, plan6 = planning_ur5e.pose_plan_path(object_pose=midpoint, approach_direction="horizon")
        planning_trajectory[i].append(plan6)
        
        # i = input()
        # if i == "end":
        #     break
        # input("press \"enter\" to approach place position")

        planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
        planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
        r_last_position, pose_goal, plan7 = planning_ur5e.pose_plan_path(object_pose=place_position, approach_direction="horizon")
        planning_trajectory[i].append(plan7)
    


        #input("press \"enter\" to cartesian path")

        planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
        planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
        pose_goal.position.z -= 0.1

        r_last_position, plan8=planning_ur5e.plan_cartesian_path(wpose=pose_goal)
        planning_trajectory[i].append(plan8)

        place_pose = copy_pose(pose_goal)


        #input("press \"enter\" to detach object")


        planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
        objects_co['milk'].mesh_poses[0] = place_pose
        planning_scene_1.detach_object(objects_co['milk'])
        plan9 = "open"
        planning_trajectory[i].append(plan9)
        
        planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)

        #input("press \"enter\" to retreat")

        pose_goal.position.x -= 0.1
        r_last_position, plan10 = planning_ur5e.plan_cartesian_path(wpose=pose_goal)
        planning_trajectory[i].append(plan10)

        #input("pree \"enter\" to plan to neutral pose")


        planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
        planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
        r_last_position, plan11 = planning_ur5e.position_plan_path(neutral_position)
        planning_trajectory[i].append(plan11)

        #input("press \"enter\" to set current pose to neutral pose")

        planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
        planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
        
        
        # compute feature mapping 
        traj_path = planning_trajectory[i]
        features_sum = get_feature_map.get_feature(objects_co=objects_co ,planning_trajectory=planning_trajectory[i])   
        
        if i == 0:
            print('end_effectors height/ ' + 'distance between eef and laptop/ ' + 'moving distance/ ' + 'distance between eef and user')
            
        print('trajectory\'s feature map =' + str(features_sum))
        
        
        
        
    planning_trajectory=np.array(planning_trajectory, dtype=object)
    np.savez("./sampled_trajectories/planning_trajectory.npz" , plan=planning_trajectory)
        
    
    
if __name__ == "__main__":
    

    

    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-trajectories", type=int, default=100,
        help="# of sampled trajectories")
    args = vars(ap.parse_args())
    main()