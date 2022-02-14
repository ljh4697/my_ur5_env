from moveit_commander.robot import RobotCommander
import rospy
import control_planning_scene
import ur5e_plan
import ur5e_get_fk
import numpy as np
from sensor_msgs.msg import JointState

from test_mesh_pickandplace import create_environment


def main():
    
    
    rospy.init_node("test_fk", anonymous=False)
    rc = RobotCommander()
    planning_scene_1 = control_planning_scene.control_planning_scene()
    planning_ur5e = ur5e_plan.ur5e_plan()
    getfk = ur5e_get_fk.GetFK('ee_link', 'world')
    env, grasp_point, approach_direction, objects_co, neutral_position = create_environment(planning_scene_1)
    
    object_position = np.zeros(3)
    object_position[0] = objects_co['milk'].mesh_poses[0].position.x
    object_position[1] = objects_co['milk'].mesh_poses[0].position.y
    object_position[2] = objects_co['milk'].mesh_poses[0].position.z
    print(object_position)
    
    ur5e_js = planning_scene_1.get_current_joint_state()
    
    planning_scene_1.get_planning_scene.robot_state.joint_state.position = np.zeros(12)
    
    
    pick_trajectories = np.load('./sampled_trajectories/pick_trajectories.npz', allow_pickle=True)
    
    for step in pick_trajectories['plan']:
        if type(step) == list:
            for traj in step:
                joint_position = np.zeros(12)
                joint_position[:6] = traj
                planning_scene_1.get_planning_scene.robot_state.joint_state.position = joint_position
                eef_position = getfk.get_fk(ur5e_js)
                distance = np.linalg.norm(eef_position-object_position)
                print(distance)
        if type(step) == str:
            print(step)
            
    position = getfk.get_fk(ur5e_js)
    print(getfk.get_fk(ur5e_js))
    
    # "ee_link"

    
if __name__ == "__main__":
    main()