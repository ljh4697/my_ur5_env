from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion
import numpy as np
from math import pi
import rospy
from moveit_commander import move_group
from moveit_commander.robot import RobotCommander
import moveit_commander
import control_planning_scene
from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion


rospy.init_node("tutorial_ur5e", anonymous=True)
robot = RobotCommander()
arm_move_group = moveit_commander.MoveGroupCommander("manipulator")
hand_move_group = moveit_commander.MoveGroupCommander("endeffector")
eef_link = arm_move_group.get_end_effector_link()
touch_links = robot.get_link_names(group="endeffector")

cs = control_planning_scene.control_planning_scene()

print(cs.get_current_joint_state())
dd = np.array([arm_move_group.get_current_pose().pose.orientation.x, 
                arm_move_group.get_current_pose().pose.orientation.y,
                arm_move_group.get_current_pose().pose.orientation.z,
                arm_move_group.get_current_pose().pose.orientation.w])
print(euler_from_quaternion(dd))

# print(hand_move_group.get_joints())
# print(hand_move_group.get_active_joints())
# print(hand_move_group.get_current_joint_values())


#print(robot.robotiq_85_left_knuckle_joint.min_bound())
#robot.robotiq_85_left_knuckle_joint.move(0.80285)
#robot.robotiq_85_left_knuckle_joint.plan(0.0)
