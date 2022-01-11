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


'''
중간 포인트를 거쳐서 pick and place 하니까 RRT motion planning 의 특징인 
unpredictable, unlegible 한 모션이 많이 planning 됨 (가끔은 최단경로와 같은 trajectory가 나옴)
따라서 같은 goal position 으로 여러번 planning 하고 가장 적은 way points를 갖는 trajectory를 select 하는 과정을 여러  position 을 이용하면 될 거 같음.



'''

def main():
    
    planning_trajectory = list()
    
    rospy.init_node("tutorial_ur5e", anonymous=True)
    robot = RobotCommander()
    # arm_move_group = moveit_commander.MoveGroupCommander("ur5e_arm")
    # arm_move_group.set_start_state_to_current_state()


    planning_scene_1 = control_planning_scene.control_planning_scene()
    planning_ur5e = ur5e_plan.ur5e_plan()

    #env, box_position, obejct_01, neutral_position = 
    env, grasp_point, approach_direction, objects_co, neutral_position = create_environment(planning_scene_1)


    approach_position ={}
    approach_position['milk'] = set_approach_position(approach_direction['milk'], grasp_point['milk'])
    midpoint = grasp_point['milk'].copy()
    place_position = grasp_point['milk'].copy()
    
    # x 0.03~0.17 y = 0.35~0.7 , z 0.03
    #midpoint[0] += 0.03 ; midpoint[1] += 0.6 ; midpoint[2] += 0.15
    midpoint[0] += 0.1 ; midpoint[1] += 0.5 ; midpoint[2] += 0.03
    
    place_position[0] += 0.05 ; place_position[1] += 1.0 ; place_position[2] += 0.1
    
    revolute_degree(approach_direction['milk'])

    r_last_position, pose_goal, plan = planning_ur5e.pose_plan_path(object_pose=approach_position['milk'], approach_direction=revolute_degree(approach_direction['milk']))
    #input("press \"enter\" to open gripper")

    planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
    planning_scene_1.r_open_gripper()
    planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)

    #input("press \"enter\" to cartesian path")


    v = direction_distance(approach_direction['milk'])
    pose_goal.position.x += v[0]
    pose_goal.position.y += v[1]
    pose_goal.position.z += v[2]
    

    r_last_position, plan=planning_ur5e.plan_cartesian_path(wpose=pose_goal)

    #print(plan)

    #input("press \"enter\" to grasp and attach box to gripper")
    
    planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
    planning_scene_1.r_close_gripper()
    planning_scene_1.attach_object(objects_co['milk'])
    planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)

    #input("press \"enter\" to retreat")

    planning_scene_1.r_open_gripper()
    planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
    pose_goal.position.z += 0.1
    
    r_last_position, plan=planning_ur5e.plan_cartesian_path(wpose=pose_goal)

    #input("press \"enter\" to approach midpoint position")

    planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
    planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
    r_last_position, pose_goal, plan = planning_ur5e.pose_plan_path(object_pose=midpoint, approach_direction="horizon")

    input("press \"enter\" to approach place position")

    planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
    planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
    r_last_position, pose_goal, plan = planning_ur5e.pose_plan_path(object_pose=place_position, approach_direction="horizon")

    input("press \"enter\" to cartesian path")

    planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
    planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
    pose_goal.position.z -= 0.1

    r_last_position, plan=planning_ur5e.plan_cartesian_path(wpose=pose_goal)

    place_pose = copy_pose(pose_goal)


    input("press \"enter\" to detach object")


    planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
    objects_co['milk'].mesh_poses[0] = place_pose
    planning_scene_1.detach_object(objects_co['milk'])
    planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)

    input("press \"enter\" to retreat")

    pose_goal.position.x -= 0.1
    r_last_position, plan=planning_ur5e.plan_cartesian_path(wpose=pose_goal)

    input("pree \"enter\" to plan to neutral pose")


    planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
    planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
    r_last_position, plan =planning_ur5e.position_plan_path(neutral_position)

    input("press \"enter\" to set current pose to neutral pose")

    planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
    planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)


if __name__ == "__main__":
    main()