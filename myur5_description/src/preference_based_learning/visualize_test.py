import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import trajectory_msgs.msg
from test_mesh_pickandplace import create_environment
import control_planning_scene




def main():
    pass



def get_user_feedback(psi, idx):




    # simulation_object.feed(input_A)
    # phi_A = simulation_object.get_features()
    # simulation_object.feed(input_B)
    # phi_B = simulation_object.get_features()
    # psi = np.array(phi_A) - np.array(phi_B)



    planning_scene_1 = control_planning_scene.control_planning_scene()

    env, grasp_point, approach_direction, objects_co, neutral_position = create_environment(planning_scene_1)

    display_trajectory_data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/sampled_trajectories/test_display_trajectory.npz', allow_pickle=True)
    start_trajectory_data = np.load('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/sampled_trajectories/test_trajectory_start.npz', allow_pickle=True)


    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("move_group_python_test", anonymous=True)

    robot = moveit_commander.RobotCommander()
    mv_group = moveit_commander.MoveGroupCommander("manipulator")

    eef_link = mv_group.get_end_effector_link()
    touch_links = robot.get_link_names(group="hand")





    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=3)

    planning_scene_1.remove_object(objects_co['milk'])
    planning_scene_1._update_planning_scene(planning_scene_1.get_planning_scene)

    s = 0


    while s == 0:



        selection = input('A/B to watch, 1/2 to vote: ').lower()

        if selection == 'a':

            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.model_id = 'ur5'
            display_trajectory.trajectory.append(display_trajectory_data['plan'][idx[0]][0])
            display_trajectory.trajectory.append(display_trajectory_data['plan'][idx[0]][1])
            display_trajectory.trajectory.append(display_trajectory_data['plan'][idx[0]][2])
            display_trajectory.trajectory.append(display_trajectory_data['plan'][idx[0]][3])

            display_trajectory.trajectory_start=start_trajectory_data['plan'][idx[0]][0]

            attached_co = moveit_msgs.msg.AttachedCollisionObject()

            attached_co.object = objects_co['milk']
            attached_co.link_name = eef_link
            attached_co.touch_links = touch_links
            display_trajectory.trajectory_start.attached_collision_objects.append(attached_co)

            for i in range(4):
                display_trajectory_publisher.publish(display_trajectory)
                rospy.sleep(0.1)

        elif selection == 'b':

            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.model_id = 'ur5'
            display_trajectory.trajectory.append(display_trajectory_data['plan'][idx[1]][0])
            display_trajectory.trajectory.append(display_trajectory_data['plan'][idx[1]][1])
            display_trajectory.trajectory.append(display_trajectory_data['plan'][idx[1]][2])
            display_trajectory.trajectory.append(display_trajectory_data['plan'][idx[1]][3])

            display_trajectory.trajectory_start=start_trajectory_data['plan'][idx[1]][0]

            attached_co = moveit_msgs.msg.AttachedCollisionObject()

            attached_co.object = objects_co['milk']
            attached_co.link_name = eef_link
            attached_co.touch_links = touch_links
            display_trajectory.trajectory_start.attached_collision_objects.append(attached_co)

            for i in range(4):
                display_trajectory_publisher.publish(display_trajectory)
                rospy.sleep(0.1)

        elif selection == '1':
            s = 1
        elif selection == '2':
            s = -1

    return psi, s








if __name__ == "__main__":
    print(get_user_feedback("dd", (0, 1)))



