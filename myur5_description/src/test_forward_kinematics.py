from moveit_commander.robot import RobotCommander
import rospy
import control_planning_scene
import ur5e_plan
import ur5e_get_fk
import numpy as np
from sensor_msgs.msg import JointState



def main():
    
    
    rospy.init_node("test_fk", anonymous=False)
    rc = RobotCommander()
    planning_scene_1 = control_planning_scene.control_planning_scene()
    planning_ur5e = ur5e_plan.ur5e_plan()
    
    print(rc.get_link_names())
    
    ur5e_js = planning_scene_1.get_current_joint_state()
    
    planning_scene_1.get_planning_scene.robot_state.joint_state.position = np.zeros(12)
    print(ur5e_js)
    
    
    pick_trajectories = np.load('./sampled_trajectories/pick_trajectories.npz', allow_pickle=True)

    getfk = ur5e_get_fk.GetFK('ee_link', 'world')
    print(getfk.get_fk(ur5e_js))
    
    # "ee_link"

def main2():
    rospy.init_node("test_fk", anonymous=False)
    rc = RobotCommander()
    planning_scene_1 = control_planning_scene.control_planning_scene()
    planning_ur5e = ur5e_plan.ur5e_plan()
    
    ur5e_js = planning_scene_1.get_current_joint_state()
    
    planning_scene_1.get_planning_scene.robot_state.joint_state.position = np.zeros(12)
    print(type(ur5e_js.name))
    jj = JointState()
    jj.name = ur5e_js.name
    print(ur5e_js)
    print(jj)
    
if __name__ == "__main__":
    main2()