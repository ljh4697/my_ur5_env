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
if __name__ == "__main__":
    
    rospy.init_node("tutorial_ur5e", anonymous=True)
    robot = RobotCommander()
    
    c_p_s = control_planning_scene.control_planning_scene()
    
    env, grasp_point, approach_direction, objects_co, neutral_position = create_environment(c_p_s)
    
    
    for obj in objects_co.keys():
        print(obj)
        c_p_s.remove_object(objects_co[obj])
        c_p_s._update_planning_scene(c_p_s.get_planning_scene)
        
    #neutral_pose = np.zeros(6)
    # c_p_s.set_joint_state_to_neutral_pose(neutral_pose)
    # c_p_s._update_planning_scene(c_p_s.get_planning_scene)