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


from tf.transformations import quaternion_from_euler


import control_planning_scene
import ur5e_plan






def copy_pose(pose:geometry_msgs.msg.Pose()):
    copied_pose = geometry_msgs.msg.Pose()

    copied_pose.position.x = pose.position.x+0.15
    copied_pose.position.y = pose.position.y
    copied_pose.position.z = pose.position.z



    return copied_pose



def create_environment():
    

    # create environment instance
    env = suite.make(
        env_name="ur5e_demo", # try with other tasks like "Stack" and "Door"
        robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
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


    return ros_cube_pos, obejct_01, joint_positions


if __name__ == "__main__":
    
    rospy.init_node("tutorial_ur5e", anonymous=True)



    planning_scene_1 = control_planning_scene.control_planning_scene()
    planning_ur5e = ur5e_plan.ur5e_plan()
    

    box_position, obejct_01, neutral_position = create_environment()



    box_position[0]-=0.25

    place_position = box_position.copy()
    place_position[0] += 0.2 ; place_position[1] += 0.2 ; place_position[2] += 0.1
    


    r_last_position, pose_goal, plan = planning_ur5e.pose_plan_path(object_pose=box_position, approach_direction="horizon")

    # print(plan)
    # print(type(box_position))

    input("press \"enter\" to open gripper")

    planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
    planning_scene_1.r_open_gripper()
    planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)

    input("press \"enter\" to cartesian path")

    pose_goal.position.x += 0.10

    r_last_position, plan=planning_ur5e.plan_cartesian_path(wpose=pose_goal)

    #print(plan)

    input("press \"enter\" to grasp and attach box to gripper")
    
    planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
    planning_scene_1.r_close_gripper()
    planning_scene_1.attach_object(obejct_01)
    planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)

    input("press \"enter\" to retreat")

    planning_scene_1.r_open_gripper()
    planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
    pose_goal.position.z += 0.1
    
    r_last_position, plan=planning_ur5e.plan_cartesian_path(wpose=pose_goal)



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
    obejct_01.primitive_poses[0] = place_pose
    planning_scene_1.detach_object(obejct_01)
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
