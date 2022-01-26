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
import ur5e_get_fk




if __name__ == "__main__":
    
    rospy.init_node("tutorial_ur5e", anonymous=True)

    pick_trajectories = np.load('./sampled_trajectories/pick_trajectories.npz', allow_pickle=True)

    
    print(np.shape(pick_trajectories['plan']))
    print(type(pick_trajectories['plan'][0]))
    print(type(pick_trajectories['plan'][0][0]))
    
    print(type(pick_trajectories['plan'][1]))
    