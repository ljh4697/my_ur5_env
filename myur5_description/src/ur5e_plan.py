#!/usr/bin/env python

import numpy as np
import trimesh
import os
import copy
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




class ur5e_plan(object):
    def __init__(self):
        super(ur5e_plan, self).__init__()
        #rospy.init_node("baxter_plan", anonymous=False)

        self.robot = RobotCommander()
    def pose_plan_path(self, object_pose, object_orientation=(0, pi, 0), arm ="manipulator" ,approach_direction="vertical"):
        


        #roscpp_initialize(sys.argv)


    #     display_trajectory_publisher = rospy.Publisher(
    #     "/move_group/display_planned_path",
    #     moveit_msgs.msg.DisplayTrajectory,
    #     queue_size=20,
    # )

        
        arm_move_group = moveit_commander.MoveGroupCommander(arm)
        arm_move_group.set_planner_id("RRTstar")
        
        eef = arm_move_group.get_end_effector_link()

        # set start state
        arm_move_group.set_start_state_to_current_state()

        # set goal state


        pose_goal = geometry_msgs.msg.Pose()
        arm_move_group.set_goal_position_tolerance(0.0001)

        if approach_direction == "horizon":
            q = quaternion_from_euler(pi/2, 0, 0,) # horizon direction approach
        elif approach_direction == "vertical":
            q = quaternion_from_euler(pi, pi/2, -pi/2) # vertical direction approach
        else:
            q = quaternion_from_euler(pi/2, 0, 0+approach_direction)
            
        


        pose_goal.orientation.x = q[0]
        pose_goal.orientation.y = q[1]
        pose_goal.orientation.z = q[2]
        pose_goal.orientation.w = q[3]

        pose_goal.position.x = object_pose[0]
        pose_goal.position.y = object_pose[1]
        pose_goal.position.z = object_pose[2]


        arm_move_group.set_pose_target(pose_goal)


        #plan = right_arm.go(wait=True)
        plan = arm_move_group.plan()

        # plan[1] = RobotTrajectory Message
        last_position = np.array(plan[1].joint_trajectory.points[-1].positions)        

        
        #display_trajectory.trajectory.append(plan[1])


        arm_move_group.stop()
        arm_move_group.clear_pose_targets()

        plan_positions = []
        for i in range(len(plan[1].joint_trajectory.points)):
            plan_positions.append(plan[1].joint_trajectory.points[i].positions)
        # Publish
        #d.publish(display_trajectory)
        return last_position, pose_goal, plan_positions


    def position_plan_path(self, desired_position, object_orientation=(0, pi, 0), arm ="manipulator"):

        arm_move_group = moveit_commander.MoveGroupCommander(arm)

        # set start state
        arm_move_group.set_start_state_to_current_state()

        # set goal state
        arm_move_group.set_goal_position_tolerance(0.001)

        arm_move_group.set_joint_value_target(desired_position)


        plan = arm_move_group.plan()

        last_position = np.array(plan[1].joint_trajectory.points[-1].positions)        

        #display_trajectory.trajectory.append(plan[1])

        arm_move_group.stop()
        arm_move_group.clear_pose_targets()

        plan_positions = []
        for i in range(len(plan[1].joint_trajectory.points)):
            plan_positions.append(plan[1].joint_trajectory.points[i].positions)
        # Publish
        #d.publish(display_trajectory)
        return last_position, plan_positions



    def plan_cartesian_path(self, wpose, scale=1, arm="manipulator"):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        arm_move_group = moveit_commander.MoveGroupCommander(arm)
        ## BEGIN_SUB_TUTORIAL plan_cartesian_path
        ##
        ## Cartesian Paths
        ## ^^^^^^^^^^^^^^^
        ## You can plan a Cartesian pa6th directly by specifying a list of waypoints
        ## for the end-effector to go through. If executing  interactively in a
        ## Python shell, set scale = 1.0.
        ##
        waypoints = []

        #wpose = arm_move_group.get_current_pose().pose
        #wpose.position.z -= scale * 0.05  # First move up (z)

        #waypoints.append(copy.deepcopy(wpose))
        waypoints.append(wpose)

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        # for this tutorial.
        (plan, fraction) = arm_move_group.compute_cartesian_path(
            waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
        )  # jump_threshold

        last_position = np.array(plan.joint_trajectory.points[-1].positions)        

        plan_positions = []
        for i in range(len(plan.joint_trajectory.points)):
            plan_positions.append(plan.joint_trajectory.points[i].positions)

        # Note: We are just planning, not asking move_group to actually move the robot yet:
        return last_position, plan_positions
    
    
    
    def plan_path(self, desired_position, arm ="manipulator"):
    
        arm_move_group = moveit_commander.MoveGroupCommander(arm)

        # set start state
        arm_move_group.set_start_state_to_current_state()

        # set goal state
        arm_move_group.set_goal_position_tolerance(0.001)

        arm_move_group.remember_joint_values(values = desired_position)


        plan = arm_move_group.plan()

        last_position = np.array(plan[1].joint_trajectory.points[-1].positions)        

        #display_trajectory.trajectory.append(plan[1])

        arm_move_group.stop()
        arm_move_group.clear_pose_targets()

        plan_positions = []
        for i in range(len(plan[1].joint_trajectory.points)):
            plan_positions.append(plan[1].joint_trajectory.points[i].positions)
        # Publish
        return last_position, plan_positions
