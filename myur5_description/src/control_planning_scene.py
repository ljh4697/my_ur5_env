#!/usr/bin/env python

import numpy as np
import trimesh
import copy
from moveit_commander import move_group
from moveit_commander.robot import RobotCommander
from moveit_commander.roscpp_initializer import roscpp_initialize, roscpp_shutdown
import rospy
import moveit_msgs.srv
import moveit_msgs.msg
import geometry_msgs.msg
from shape_msgs.msg import SolidPrimitive, Plane, Mesh, MeshTriangle
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene, ApplyPlanningSceneRequest
from geometry_msgs.msg import Point
from moveit_msgs.srv import GetPlanningScene, GetPlanningSceneRequest
from moveit_msgs.msg import Grasp
from math import pi, tau
import moveit_commander
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

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))



class control_planning_scene(object):
    def __init__(self):
        super(control_planning_scene, self).__init__()

        #rospy.init_node("plannig_scene" , anonymous=False)

        self.robot = moveit_commander.RobotCommander()


        self.get_ps_srv = self._get_planning_response()
        self.get_planning_scene = self._get_planning_response_call(self.get_ps_srv).scene
        self.apply_ps_srv = rospy.ServiceProxy('apply_planning_scene', ApplyPlanningScene)


    def _get_planning_response(self):
        get_ps_srv = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene())
        return get_ps_srv

    def _get_planning_response_call(self, get_ps_srv):
        get_req_ = GetPlanningSceneRequest()
        get_ps_srv.wait_for_service(0.5)
        return get_ps_srv.call(get_req_)

    def _make_mesh(self, name, pose, mesh_path, size=(1, 1, 1)):
        co = CollisionObject()
        co.operation = CollisionObject.ADD
        co.id = name
        co.header = pose.header

        #make mesh
        mesh = trimesh.load(mesh_path, force='mesh')

        mesh_01 = Mesh()
        for face in mesh.faces:
            triangle = MeshTriangle()
            triangle.vertex_indices = face
            mesh_01.triangles.append(triangle)

        for vertex in mesh.vertices:
            point = Point()
            point.x = vertex[0] * size[0]
            point.y = vertex[1] * size[1]
            point.z = vertex[2] * size[2]
            mesh_01.vertices.append(point)
        
        co.meshes = [mesh_01]
        co.mesh_poses = [pose.pose]

        self.get_planning_scene.world.collision_objects.append(co)

        return co


    def _make_box(self, name, pos, quat=(0, 0, 0, 0), size= (0.1, 0.1, 0.1), header="world"):

        object_pose = geometry_msgs.msg.PoseStamped()

        object_pose.header.frame_id = header

        object_pose.pose.position.x = pos[0]
        object_pose.pose.position.y = pos[1]
        object_pose.pose.position.z = pos[2]
        object_pose.pose.orientation.x = quat[0]
        object_pose.pose.orientation.y = quat[1]
        object_pose.pose.orientation.z = quat[2]
        object_pose.pose.orientation.w = quat[3]

        co = CollisionObject()
        co.id = name
        co.operation = CollisionObject.ADD
        co.header = object_pose.header

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = list(size)

        co.primitives = [box]
        co.primitive_poses = [object_pose.pose]

        self.get_planning_scene.world.collision_objects.append(co)

        return co

    def _make_cylinder(self, name, pos, size, quat=(0, 0, 0, 0), header="world"):


        object_pose = geometry_msgs.msg.PoseStamped()

        object_pose.header.frame_id = header

        object_pose.pose.position.x = pos[0]
        object_pose.pose.position.y = pos[1]
        object_pose.pose.position.z = pos[2]
        object_pose.pose.orientation.x = quat[0]
        object_pose.pose.orientation.y = quat[1]
        object_pose.pose.orientation.z = quat[2]
        object_pose.pose.orientation.w = quat[3]

        co = CollisionObject()
        co.id = name
        co.operation = CollisionObject.ADD
        co.header = object_pose.header

        cylinder = SolidPrimitive()
        cylinder.type = SolidPrimitive.CYLINDER
        cylinder.dimensions = list(size)

        co.primitives = [cylinder]
        co.primitive_poses = [object_pose.pose]

        self.get_planning_scene.world.collision_objects.append(co)
        return co
        
        return

    def _update_planning_scene(self, ps:PlanningScene):
        ps.is_diff = True   
        ps.robot_state.is_diff = True

        apply_req = ApplyPlanningSceneRequest()
        apply_req.scene = ps

        self.apply_ps_srv.call(apply_req)

        return

    def remove_entire_objects(self):
        pass

    def get_current_joint_state(self):
        return self.get_planning_scene.robot_state.joint_state.position

    def set_joint_state_to_neutral_pose(self, neutral_pose=[0]):
        current_position = np.array(self.get_planning_scene.robot_state.joint_state.position)
        current_position[:6] = neutral_pose
        self.get_planning_scene.robot_state.joint_state.position = current_position
       

    def get_joint(self):
        pass


    def r_open_gripper(self):
        current_pose = np.array(self.get_planning_scene.robot_state.joint_state.position)
        current_pose[6:] = np.array([0, 0, 0, 0, 0, 0])

        self.get_planning_scene.robot_state.joint_state.position = current_pose
        return
    
    def r_close_gripper(self):
        current_pose = np.array(self.get_planning_scene.robot_state.joint_state.position)
        current_pose[6:] = np.array([0.8027538274584142, -0.8027538274584142, 0.8027538274584142, 0.8027538274584142, -0.8027538274584142, 0.8027538274584142])

        self.get_planning_scene.robot_state.joint_state.position = current_pose
        return

    
    def attach_object(self, _object, grasping_group="hand"):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        robot = self.robot
        
        arm = moveit_commander.MoveGroupCommander("manipulator")
        eef_link = arm.get_end_effector_link()
        
        touch_links = robot.get_link_names(group=grasping_group)

        remove_obejct = CollisionObject()
        remove_obejct.id = _object.id
        remove_obejct.header.frame_id = _object.header.frame_id
        remove_obejct.operation = remove_obejct.REMOVE

        aco = AttachedCollisionObject()

        aco.object.id = _object.id
        aco.object.header.frame_id = eef_link
        aco.link_name = eef_link
        aco.touch_links = touch_links
        aco.object.operation = aco.object.ADD



        # scene = PlanningScene()
        # scene.is_diff = True
        # scene.robot_state.is_diff = True



        self.get_planning_scene.robot_state.attached_collision_objects.clear()
        self.get_planning_scene.world.collision_objects.clear()
        
        self.get_planning_scene.world.collision_objects.append(remove_obejct)
        self.get_planning_scene.robot_state.attached_collision_objects.append(aco)
        # We wait for the planning scene to update.
        return 
    

    def detach_object(self, _object, grasping_group="hand"):

        robot = self.robot


        arm = moveit_commander.MoveGroupCommander("manipulator")
        eef_link = arm.get_end_effector_link()

        touch_links = robot.get_link_names(group=grasping_group)

        detach_object = AttachedCollisionObject()

        detach_object.object.id = _object.id
        detach_object.object.header.frame_id = eef_link
        detach_object.link_name = eef_link
        detach_object.touch_links = touch_links
        detach_object.object.operation = detach_object.object.REMOVE


        re_introduce_object = CollisionObject()
        re_introduce_object = _object


        self.get_planning_scene.robot_state.attached_collision_objects.clear()
        self.get_planning_scene.world.collision_objects.clear()

        self.get_planning_scene.robot_state.attached_collision_objects.append(detach_object)
        self.get_planning_scene.world.collision_objects.append(re_introduce_object)
        return 