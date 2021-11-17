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




def copy_pose(pose:geometry_msgs.msg.Pose()):
    copied_pose = geometry_msgs.msg.Pose()

    copied_pose.position.x = pose.position.x
    copied_pose.position.y = pose.position.y
    copied_pose.position.z = pose.position.z



    return copied_pose



def create_environment():
    
    config = load_controller_config(default_controller="JOINT_POSITION")
    # create environment instance
    env = suite.make(
        env_name="ur5e_demo", # try with other tasks like "Stack" and "Door"
        robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
        controller_configs=config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    # reset the environment
    env.reset()


    object_size = (env.ros_cube_size[0]*2, env.ros_cube_size[1]*2, env.ros_cube_size[2]*2)
    
    table_visual_size = (env.ros_table_visual_size[0]*2 , env.ros_table_visual_size[1]*2, env.ros_table_visual_size[2]*2)

    table_legs_size = (env.ros_table_legs_size[1]*2 ,env.ros_table_legs_size[0])
    table_legs_pos = env.ros_table_legs_pos

    ros_cube_pos = env.ros_cube_pos
    ros_table_visual_pos = env.ros_table_visual_pos
    
    ros_table_visual_pos[2] -= 0


    for i in range(4):
        table_legs_pos[i][2] -= 0
        
        
    # add box
    obejct_01 = planning_scene_1._make_box(name="object", pos=ros_cube_pos, quat=env.ros_cube_quaternion, size = object_size)
    # add table (top)
    planning_scene_1._make_box(name="table_visual", pos=ros_table_visual_pos, size = table_visual_size)
    # add table legs
    for i in range(4):
        planning_scene_1._make_cylinder(name=f"table_legs{i}", pos=table_legs_pos[i], size=table_legs_size)


    joint_positions = env.robots[0]._joint_positions
    # set start pose to neutral pose 

    planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=joint_positions)

    # apply planning scene
    planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)





    # for i in range(1000):
    #     action = np.random.randn(env.robots[0].dof) # sample random action
    #     obs, reward, done, info = env.step(action)  # take action in the environment
    #     env.render()  # render on display


    return env, ros_cube_pos, obejct_01, joint_positions


if __name__ == "__main__":

    rospy.init_node("tutorial_ur5e", anonymous=True)
    robot = RobotCommander()
    # arm_move_group = moveit_commander.MoveGroupCommander("ur5e_arm")
    # arm_move_group.set_start_state_to_current_state()


    planning_scene_1 = control_planning_scene.control_planning_scene()
    planning_ur5e = ur5e_plan.ur5e_plan()

    env, box_position, obejct_01, neutral_position = create_environment()



    goal = box_position.copy()
    goal[2]+=0.18
    r_last_position, pose_goal, plan = planning_ur5e.pose_plan_path(object_pose=goal)
    
    
    place_position = box_position.copy()
    place_position[0] += 0.1 ; place_position[1] += 0.2


    #input()



    desired_positions = plan

    i = 0
    delta_q = np.zeros(7)
    goal_reach = False
    done = False
    env.viewer.set_camera(camera_id=0)

    while True:
        
        
        s = 1
        
        while not goal_reach:
            current_positions = env.robots[0]._joint_positions
            delta_q[:6] = desired_positions[i]-current_positions
            action = delta_q.copy()
            action *= (s/100)
                
            #check joint position
            #env.robots[0].sim.data.qpos[:6] = desired_positions[i]


            #obs, reward, done, info = env.step(action)  # take action in the environment
            env.step(action)
            env.render()  # render on display



            goal_reach = True
            print(delta_q, 'step', i, 's = ', action.sum(), s)
            if i < len(desired_positions)-1:
                for e in range(len(delta_q)):
                    if np.abs(delta_q[e]) > 0.01:
                        goal_reach = False
                        break
            else:
                print("end")
                for e in range(len(delta_q)):
                    if np.abs(delta_q[e]) > 0.003:
                        goal_reach = False
                        break
                
            s += 4
            #print(env.robots[0]._hand_pos)


        goal_reach = False

        if i < len(desired_positions)-1:
            i+=1
        else:
            done = True
