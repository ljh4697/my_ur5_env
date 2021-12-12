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
        env_name="ur5e_pickandplace", # try with other tasks like "Stack" and "Door"
        robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
        controller_configs=config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    
    env.reset()
    
    
    print(env.ros_objects_pos)



    for i in range(5):
        print(env.ros_objects_pos)

        action=np.zeros(7)
        env.step(action)
        env.render()  # render on display

    



        
    meshes_path = {}
    
    for name in env.obj_names:
        meshes_path[name] = "/home/joonhyeok/robosuite/robosuite/models/assets/objects/meshes/"+ name +".stl"
        
    print(meshes_path)


    # reset the environmentplanning_scene_1


    object_size = (env.ros_cube_size[0]*2, env.ros_cube_size[1]*2, env.ros_cube_size[2]*2)

    table_visual_size = (env.ros_table_visual_size[0]*2 , env.ros_table_visual_size[1]*2, env.ros_table_visual_size[2]*2)

    table_legs_size = (env.ros_table_legs_size[1]*2 ,env.ros_table_legs_size[0])
    table_legs_pos = env.ros_table_legs_pos

    #ros_cube_pos = env.ros_cube_pos
    
    ros_objects_pos = env.ros_objects_pos
    ros_objects_quaternion = env.ros_objects_quaternion
    
    
    ros_table_visual_pos = env.ros_table_visual_pos
    
    ros_table_visual_pos[2] -= 0


    for i in range(4):
        table_legs_pos[i][2] -= 0
        
        
    # add box
    #obejct_01 = planning_scene_1._make_box(name="object", pos=ros_cube_pos, quat=env.ros_cube_quaternion, size = object_size)
    # add table (top)
    for i, n in enumerate(env.obj_names):
        co_i = planning_scene_1._make_mesh(name =n, mesh_path=meshes_path[n], pos=ros_objects_pos[n] ,quat=ros_objects_quaternion[n])
        
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


    return env


if __name__ == "__main__":

    rospy.init_node("tutorial_ur5e", anonymous=True)
    robot = RobotCommander()
    # arm_move_group = moveit_commander.MoveGroupCommander("ur5e_arm")
    # arm_move_group.set_start_state_to_current_state()


    planning_scene_1 = control_planning_scene.control_planning_scene()
    planning_ur5e = ur5e_plan.ur5e_plan()

    #env, box_position, obejct_01, neutral_position = 
    env = create_environment()







    desired_positions = [(-0.4865757957823134, -1.7595564350026254, 2.4746010722084986, -2.261700043048569, -1.6014259813153882, -1.99319855190651), (-0.4054703427207998, -1.531573823535642, 2.3238865383867173, -2.346935685343888, -1.5914791494325642, -1.9336917610246314), (-0.32436488965928617, -1.3035912120686588, 2.173172004564936, -2.4321713276392067, -1.58153231754974, -1.8741849701427526), (-0.24325943659777255, -1.0756086006016754, 2.022457470743155, -2.517406969934526, -1.571585485666916, -1.814678179260874)]
    i = 0

    delta_q = np.zeros(7)
    goal_reach = False
    while True:

        
        while not goal_reach:

                    
            while True:
                action=np.zeros(7)
                env.step(action)
                env.render()  # render on display
