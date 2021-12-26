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



'''
vertical 처럼 approach 할때 distance 줬던거 처럼 똑같은 distanch 로 approach 하면 안됨
approach 하기 전에 distance 준 것 보다 작은 distanch 로 approach 해야 됨.


dynamic world 에서 무거운 물체 들 때 로봇팔이 흔들리는걸 parameter 튜닝해서 고쳐보자.

action *= (s/100) 이 텀이 없으면 애초에 무거운 물체를 들지도 못한다 (PID 에서 필요한 토크보다 정수배 해서 토크를 늘려야 함).
무거운 물체를 들때는 그 물체의 무게를 고려한 중력 보상텀이 따로 필요해 보인다.


'''



def copy_pose(pose:geometry_msgs.msg.Pose()):
    copied_pose = geometry_msgs.msg.Pose()

    copied_pose.position.x = pose.position.x+0.15
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
    
    



    for i in range(5):

        action=np.zeros(7)
        env.step(action)
        env.render()  # render on display

    



        
    meshes_path = {}
    
    for name in env.obj_names:
        meshes_path[name] = "/home/joonhyeok/robosuite/robosuite/models/assets/objects/meshes/"+ name +".stl"
        


    # reset the environmentplanning_scene_1


    #object_size = (env.ros_cube_size[0]*2, env.ros_cube_size[1]*2, env.ros_cube_size[2]*2)

    table_visual_size = (env.ros_table_visual_size[0]*2 , env.ros_table_visual_size[1]*2, env.ros_table_visual_size[2]*2)

    table_legs_size = (env.ros_table_legs_size[1]*2 ,env.ros_table_legs_size[0])
    table_legs_pos = env.ros_table_legs_pos

    #ros_cube_pos = env.ros_cube_pos
    
    ros_objects_pos = env.ros_objects_pos
    ros_objects_quaternion = env.ros_objects_quaternion
    objects_co = {}
    
    
    ros_table_visual_pos = env.ros_table_visual_pos
    
    ros_table_visual_pos[2] -= 0


    for i in range(4):
        table_legs_pos[i][2] -= 0
        
        
    # add box
    #obejct_01 = planning_scene_1._make_box(name="object", pos=ros_cube_pos, quat=env.ros_cube_quaternion, size = object_size)
    # add table (top)
    for i, n in enumerate(env.obj_names):
        objects_co[n] = planning_scene_1._make_mesh(name =n, mesh_path=meshes_path[n], pos=ros_objects_pos[n] ,quat=ros_objects_quaternion[n])
        
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


    return env, ros_objects_pos, objects_co, joint_positions


if __name__ == "__main__":

    rospy.init_node("tutorial_ur5e", anonymous=True)
    robot = RobotCommander()
    # arm_move_group = moveit_commander.MoveGroupCommander("ur5e_arm")
    # arm_move_group.set_start_state_to_current_state()


    planning_scene_1 = control_planning_scene.control_planning_scene()
    planning_ur5e = ur5e_plan.ur5e_plan()

    env, ros_objects_pos, objects_co, neutral_position = create_environment()



    ros_objects_pos['bottle'][0]-=0.25

    place_position = ros_objects_pos['bottle'].copy()
    place_position[0] += 0.2 ; place_position[1] += 0.2 ; place_position[2] += 0.1

    separated_task = 0


    gripper_state = ""

    # pick and place
    
    while True:
        if separated_task == 0: # approach
            r_last_position, pose_goal, plan = planning_ur5e.pose_plan_path(object_pose=ros_objects_pos['bottle'] , approach_direction="horizon")
        elif separated_task == 1: # open gripper
            
            planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
            planning_scene_1.r_open_gripper()
            gripper_state = "OPEN"
            planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)

        elif separated_task == 2: # cartesian path
            gripper_state = ""
            pose_goal.position.x += 0.1
            r_last_position, plan=planning_ur5e.plan_cartesian_path(wpose=pose_goal)
            
        elif separated_task == 3: # grasp and attach box to gripper
            planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
            planning_scene_1.r_close_gripper()
            gripper_state = "CLOSE"
            planning_scene_1.attach_object(objects_co['bottle'])
            planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
            
        elif separated_task == 4: # retreat   
            gripper_state = ""
            planning_scene_1.r_open_gripper()
            planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
            pose_goal.position.z += 0.1
            r_last_position, plan=planning_ur5e.plan_cartesian_path(wpose=pose_goal)
        
        elif separated_task == 5: #  approach place position
            gripper_state = ""
            planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
            planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
            r_last_position, pose_goal, plan = planning_ur5e.pose_plan_path(object_pose=place_position , approach_direction="horizon")
    
        elif separated_task == 6: # cartesian path
            gripper_state = ""
            planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
            planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
            pose_goal.position.z -= 0.1
            r_last_position, plan=planning_ur5e.plan_cartesian_path(wpose=pose_goal)
            place_pose = copy_pose(pose_goal)
        
        elif separated_task == 7: # detach object"
            gripper_state = "OPEN"
            planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
            objects_co['bottle'].mesh_poses[0] = place_pose
            planning_scene_1.detach_object(objects_co['bottle'])
            planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
        
        elif separated_task == 8: # to retreat
            gripper_state = ""
            pose_goal.position.x -= 0.1
            r_last_position, plan=planning_ur5e.plan_cartesian_path(wpose=pose_goal)
        
        elif separated_task == 9: # set current pose to neutral pose
            gripper_state = ""
            planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
            planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
            r_last_position, plan =planning_ur5e.position_plan_path(neutral_position)
        
        elif separated_task == 10:  # set current pose to neutral pose
            gripper_state = ""
            planning_scene_1.set_joint_state_to_neutral_pose(neutral_pose=r_last_position)
            planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)
            break
            

        
        desired_positions = plan
        
        i = 0
        delta_q = np.zeros(7)
        last_action = np.zeros(7)
        goal_reach = False
        terminate = False
        
        if gripper_state != "":
            
            
            action=np.zeros(7)
            if gripper_state == "CLOSE":
                for i in range(100):
                    action[-1] = 0.5
                    env.step(action)
                    env.render()  # render on display
            if gripper_state =="OPEN":
                for i in range(100):
                    action[-1] = -0.5
                    env.step(action)
                    env.render()  # render on display


        else:
            
            while not terminate:
                
                
                s = 1
                
                while not goal_reach:
                    
                    current_positions = env.robots[0]._joint_positions
                    delta_q[:6] = desired_positions[i]-current_positions
                    
                    
                    action = delta_q.copy()
                    action *= (s/100)
                    
                    if s <= 120:
                        last_action*=0.9
                        action += last_action
                    #obs, reward, done, info = env.step(action)  # take action in the environment
                    
                    env.step(action)
                    env.render()  # render on display



                    goal_reach = True
                    #print(delta_q, 'step', i, 's = ', action.sum(), s)
                    if i < len(desired_positions)-1:
                        for e in range(len(delta_q)):
                            if np.abs(delta_q[e]) > 0.06:
                                goal_reach = False
                                break
                    else:
                        for e in range(len(delta_q)):
                            if np.abs(delta_q[e]) > 0.003:
                                goal_reach = False
                                break
                            
                    if goal_reach == True:
                        last_action = action.copy()
                        
                    s += 5


                goal_reach = False

                if i < len(desired_positions)-1:
                    i+=1
                else:
                    terminate = True




        
        separated_task += 1
