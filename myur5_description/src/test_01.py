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
from tf.transformations import quaternion_from_euler

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
import ur5e_get_fk

from robosuite import load_controller_config
from test_mesh_pickandplace import cross_product

def create_environment(planning_scene_1):
    
    config = load_controller_config(default_controller="JOINT_POSITION")
    # create environment instance
    env = suite.make(
        env_name="ur5e_pickandplace", # try with other tasks like "Stack" and "Door"
        robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
        controller_configs=config,
        has_renderer=True,
        render_camera=None,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    
    env.reset()
    



    for i in range(5):

        action=np.zeros(7)
        env.step(action)
        env.render()  # render on display

    



        
    meshes_path = {}
    
    for name in env.obj_names:
        meshes_path[name] = "/home/joonhyeok/robosuite/robosuite/models/assets/objects/meshes/"+ name +".stl"
        

    meshes_path['user'] = '/home/joonhyeok/robosuite/robosuite/models/assets/objects/meshes/human.stl'

    meshload = {}
    approach_direction = {}
    grasp_point = {}
    
    ros_objects_pos = env.ros_objects_pos
    
    # find grasp point
    # for obj in meshes_path.keys():
    #     m = obj
    #     meshload[m]= trimesh.load(meshes_path[m], force='mesh')
    #     mesh_center_mass = meshload[m].center_mass
    #     dot0_face_z = np.where(np.dot(meshload[m].face_normals[:],np.array([0, 0, 1]))==0)[0]

    #     num_faces = len(dot0_face_z)
    #     weights = np.zeros(len(meshload[m].faces))
    #     weights[dot0_face_z] = 1/num_faces

    #     samples, face_idx = meshload[m].sample(100, return_index = True, face_weight = weights)
    #     sample_opposite_normals = np.zeros_like(samples)
    #     sample_opposite_normals = -meshload[m].face_normals[face_idx, :]

    #     mesh_centeres = np.ones_like(samples)*mesh_center_mass
        
        
    #     distances = np.linalg.norm(mesh_center_mass-samples, axis=1)
    #     sorted_distances = np.sort(distances)

        
    #     FIND_GRASP_POINTS = False
    #     iter = 0
    #     while not FIND_GRASP_POINTS: 
    #         cloeset_point = samples[np.where(distances[:] == sorted_distances[iter])]
    #         normal_vector = sample_opposite_normals[np.where(distances[:] == sorted_distances[iter])]

    #         # find rayed point
    #         #location, idx, f = mesh.ray.intersects_location(ray_origins=samples, ray_directions=sample_opposite_normals)
    #         location, idx, f = meshload[m].ray.intersects_location(ray_origins=np.array(cloeset_point), ray_directions=np.array(normal_vector))


    #         if len(idx) == 2 and round(np.dot(meshload[m].face_normals[f][0, :], meshload[m].face_normals[f][1, :]),3) == -1:
    #             FIND_GRASP_POINTS = True
    #         else:
    #             iter += 1
    #     approach_direction[m]= cross_product(location[1]-location[0], [0, 0, 1])
    #     grasp_point[m] = ros_objects_pos[m] - [0, 0, 0.0003]
        


    #enter env information
    
    #object_size = (env.ros_cube_size[0]*2, env.ros_cube_size[1]*2, env.ros_cube_size[2]*2)

    table_visual_size = (env.ros_table_visual_size[0]*2 , env.ros_table_visual_size[1]*2, env.ros_table_visual_size[2]*2)

    table_legs_size = (env.ros_table_legs_size[1]*2 ,env.ros_table_legs_size[0])
    table_legs_pos = env.ros_table_legs_pos

    #ros_cube_pos = env.ros_cube_pos
    
    ros_objects_quaternion = env.ros_objects_quaternion
    objects_co = {}
    
    
    ros_table_visual_pos = env.ros_table_visual_pos
    
    ros_table_visual_pos[2] -= 0


    for i in range(4):
        table_legs_pos[i][2] -= 0
        
        
    # add box
    #obejct_01 = planning_scene_1._make_box(name="object", pos=ros_cube_pos, quat=env.ros_cube_quaternion, size = object_size)
    # add table (top)
    q = quaternion_from_euler(pi/2, 0, -pi/2,)
    
    for i, n in enumerate(meshes_path.keys()):
        if n == "milk":
            objects_co[n] = planning_scene_1._make_mesh(name =n, mesh_path=meshes_path[n], pos=ros_objects_pos[n] ,quat=ros_objects_quaternion[n], size=(0.9, 0.9, 0.9))
        elif n == "laptop":
            objects_co[n] = planning_scene_1._make_mesh(name =n, mesh_path=meshes_path[n], pos=ros_objects_pos[n] ,quat=ros_objects_quaternion[n], size=(0.08, 0.08, 0.08))
        elif n == "user":
            objects_co[n] = planning_scene_1._make_mesh(name =n, mesh_path=meshes_path[n], pos=np.array((1.3, 0, 0)) ,quat=np.array(q), size=(0.013, 0.013, 0.013))
            
            
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


    return env, grasp_point, approach_direction, objects_co, joint_positions



if __name__ == "__main__":
    
    rospy.init_node("tutorial_ur5e", anonymous=True)
    robot = RobotCommander()
    # arm_move_group = moveit_commander.MoveGroupCommander("ur5e_arm")
    # arm_move_group.set_start_state_to_current_state()


    planning_scene_1 = control_planning_scene.control_planning_scene()
    planning_ur5e = ur5e_plan.ur5e_plan()

    #env, box_position, obejct_01, neutral_position = 
    env, grasp_point, approach_direction, objects_co, neutral_position = create_environment(planning_scene_1)


    