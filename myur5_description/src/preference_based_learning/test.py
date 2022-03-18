import numpy as np
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import trajectory_msgs.msg








def main():


    display_trajectory_data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/sampled_trajectories/test_display_trajectory.npz', allow_pickle=True)
    start_trajectory_data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/sampled_trajectories/test_trajectory_start.npz', allow_pickle=True)

    print(type(display_trajectory_data['plan'][0][0]))
    print(type(start_trajectory_data['plan'][0][0]))

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("move_group_python_test", anonymous=True)

    robot = moveit_commander.RobotCommander()
    mv_group = moveit_commander.MoveGroupCommander("manipulator")

    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=10)

    display_trajectory = moveit_msgs.msg.DisplayTrajectory()

    # for i in range(len(display_trajectory_data['plan'][0][1])):
    #     print(display_trajectory_data['plan'][0][1][i])
    #     trajectory_point = trajectory_msgs.msg.JointTrajectoryPoint()
    #     trajectory_point.positions.append(display_trajectory_data['plan'][0][1][i])
    #     trajectory.joint_trajectory.points.append(trajectory_point)
    display_trajectory.model_id = 'ur5'
    display_trajectory.trajectory.append(display_trajectory_data['plan'][0][0])
    display_trajectory.trajectory_start=start_trajectory_data['plan'][0][0]
    while not rospy.is_shutdown():
    
        display_trajectory_publisher.publish(display_trajectory)
        rospy.sleep(0.1)
        break
                                            





if __name__ == "__main__":
    main()



