import xml.etree.ElementTree as ET
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, PolygonStamped, Point
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from scipy.spatial.transform import Rotation
import numpy as np
import rospy
import copy
from transform import TF_mat
import traj_gen

import sys
sys.path.append('/home/lyh/catkin_ws/src/suhan_robot_model_tools')
from srmt.planning_scene import PlanningScene
from srmt.planner.rrt_connect import SuhanRRTConnect
from srmt.kinematics import TRACIK
from suhan_robot_model_tools.suhan_robot_model_tools_wrapper_cpp import ( 
                    vectors_to_isometry, isometry_to_vectors, isometry_product, NameVector)
import time
import moveit_commander
import moveit_msgs.msg, trajectory_msgs.msg

q_init = np.array([ 0.0, 0.0, 0.92983, 0.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, -0.24, 0.6, -0.36, 0.0, 
                    0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                    0.0, 0.0, 0.0,
                    0.3, 0.3, 1.5, -1.27, -1.0, 0.0, -1.0, 0.0,
                    0.0, 0.0,
                    -0.3, -0.3, -1.5, 1.27, 1.0, 0.0, 1.0, 0.0])
cup_pos = np.zeros(3)
pelvis_current = np.zeros(7)
q_current = np.zeros(33)

planning_scene = PlanningScene(['pelvis_pose', 'whole_body'], [7, 33], topic_name='mujoco_tocabi', base_link='/world')
robot = moveit_commander.RobotCommander()
move_group = moveit_commander.MoveGroupCommander('right_arm')

def transform_matrix(pose, inverse = False):
    p = np.array([pose.position.x, pose.position.y, pose.position.z])
    q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    R = Rotation.from_quat(q).as_matrix()
    T = np.eye(4)
    if inverse:
        T[:3,:3] = R.transpose()
        T[:3,3] = -np.matmul(R.transpose(), p)
    else:
        T[:3,:3] = R
        T[:3,3] = p

    return T

def base_converter(pose, base_pose):
    T1 = transform_matrix(pose)
    T2 = transform_matrix(base_pose, inverse=True)
    T = np.matmul(T2, T1)
    p = T[:3,3]
    q = Rotation.from_matrix(T[:3,:3]).as_quat()

    new_pose = Pose()
    new_pose.position.x = p[0]
    new_pose.position.y = p[1]
    new_pose.position.z = p[2]
    new_pose.orientation.x = q[0]
    new_pose.orientation.y = q[1]
    new_pose.orientation.z = q[2]
    new_pose.orientation.w = q[3]

    return new_pose

def pose_to_vectors(pose):
    pos = []
    quat = []

    pos.append(pose.position.x)
    pos.append(pose.position.y)
    pos.append(pose.position.z)

    quat.append(pose.orientation.x)
    quat.append(pose.orientation.y)
    quat.append(pose.orientation.z)
    quat.append(pose.orientation.w)

    return pos, quat

def vectors_to_pose(pos, quat):
    pose = Pose()

    pose.position.x = pos[0]
    pose.position.y = pos[1]
    pose.position.z = pos[2]

    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]

    return pose



def get_primitive(attribute):
    primitive = SolidPrimitive()
    if attribute['type'] == 'box':
        primitive.type = primitive.BOX
        for s in attribute['size'].split(' '):
            primitive.dimensions.append(float(s)*2)

    elif attribute['type'] == 'sphere':
        primitive.type = primitive.SPHERE
        for s in attribute['size'].split(' '):
            primitive.dimensions.append(float(s))

    elif attribute['type'] == 'cylinder':
        primitive.type = primitive.CYLINDER
        size = attribute['size'].split(' ')
        if len(size) == 1:
            primitive.dimensions.append(float(size[0]))
        else:
            primitive.dimensions.append(float(size[1])*2)   # height
            primitive.dimensions.append(float(size[0]))     # radius

    pose = Pose()
    pose.orientation.w = 1.0
    if 'pos' in attribute:
        pos = attribute['pos'].split(' ')
        pose.position.x = float(pos[0])
        pose.position.y = float(pos[1])
        pose.position.z = float(pos[2])
    if 'quat' in attribute:
        quat = attribute['quat'].split(' ')
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])

    return primitive, pose


def make_object_msg(body, msg=None):
    for child in body:
        if child.tag == 'geom':
            primitive, pose = get_primitive(child.attrib)
            if msg is None:
                msg = CollisionObject()
                msg.id = body.attrib['name']
                msg.header.frame_id = 'world'
                msg.pose = pose
                msg.operation = msg.ADD
                p = Pose()
                p.orientation.w = 1
                msg.primitive_poses.append(p)
            else:
                msg.primitive_poses.append(base_converter(pose, msg.pose))
            msg.primitives.append(primitive)

        elif child.tag == 'body':
            make_object_msg(child, msg)

    return msg


tree = ET.parse('/home/lyh/catkin_ws/src/tocabi/dyros_tocabi_v2/tocabi_description/mujoco_model/dyros_tocabi_with_object_2024winter.xml')
root = tree.getroot()
worldbody = None
for child in root:
    if child.tag == 'worldbody':
        worldbody = child

msgs = []
for body in worldbody:
    if body.tag == 'body' and not body.attrib['name'].endswith('link'):
        msg = make_object_msg(body)
        msgs.append(msg)
        planning_scene.apply_collision_object_msg(msg)

import message_filters

filename1 = 'cur_right_arm.txt'
filename2 = 'traj_right_arm.txt'
f1 = open('/home/lyh/catkin_ws/src/tocabi/tocabi_srmt/data/{}'.format(filename1), 'w')
f2 = open('/home/lyh/catkin_ws/src/tocabi/tocabi_srmt/data/{}'.format(filename2), 'w')

def save_joints(file, msg):
    file.write('{} '.format(msg.header.stamp.to_sec()))
    for q in msg.position[-8:]:
        file.write('{} '.format(q))
    for v in msg.velocity[-8:]:
        file.write('{} '.format(v))
    file.write('\n')

def save_traj(file, msg):
    for point in msg.joint_trajectory.points:
        file.write('{} '.format(point.time_from_start.to_sec()))
        for q in point.positions[-8:]:
            file.write('{} '.format(q))
        for v in point.velocities[-8:]:
            file.write('{} '.format(v))
        for a in point.accelerations[-8:]:
            file.write('{} '.format(a))
        file.write('\n')


def sync_callback(point_msg, joint_msg):
    pelvis_pos = point_msg.polygon.points[3]
    pelvis_rpy = point_msg.polygon.points[4]
    pelvis_quat = Rotation.from_euler('xyz', [pelvis_rpy.x, pelvis_rpy.y, pelvis_rpy.z]).as_quat()
    q = joint_msg.position

    global pelvis_current
    global q_current
    pelvis_current= np.array([pelvis_pos.x, pelvis_pos.y, pelvis_pos.z, *pelvis_quat])
    q_current = np.array([*q])

    q = np.concatenate([pelvis_current, q_current])

    save_joints(f1, joint_msg)

    planning_scene.display(q)

point_sub = message_filters.Subscriber("/tocabi/point", PolygonStamped)
joint_sub = message_filters.Subscriber("/tocabi/jointstates", JointState)
ts = message_filters.ApproximateTimeSynchronizer([point_sub, joint_sub], 10, 0.001)
ts.registerCallback(sync_callback)

# trajectory_pub = rospy.Publisher("/tocabi/srmt/trajectory", JointTrajectoryPoint)
trajectory_pub = rospy.Publisher("/tocabi/srmt/trajectory", JointTrajectory)

def update_cup_pos(point_msg):
    cup_pos[0] = point_msg.x
    cup_pos[1] = point_msg.y
    cup_pos[2] = point_msg.z
    planning_scene.update_object_pose("cup", cup_pos, [0, 0, 0, 1])

cup_sub = rospy.Subscriber("/cup_pos", Point, update_cup_pos)

tracik_body = TRACIK('Pelvis_Link', 'Upperbody_Link')
tracik_right = TRACIK('Upperbody_Link', 'R_Wrist2_Link')
joint_upper_limit = tracik_right.get_upper_bound()
joint_lower_limit = tracik_right.get_lower_bound()
v_max = [10.0] * tracik_right.get_num_joints()

ee2grasp = TF_mat.from_vectors([0, 0, -0.1], [-0.7071068, 0, 0.7071068, 0])

while rospy.is_shutdown() is False:
    print("==========================\n"
        "Press a key and hit Enter to execute an action.\n"
        "0 to exit\n"
        "1 to grab the cup")
    character_input = input()

    if character_input == '0':
        f1.close()
        f2.close()
        break
    
    elif character_input == '1':
        body_rel_pos, body_rel_quat = tracik_body.forward_kinematics(q_current[12:15])
        body_tf = TF_mat.mul(TF_mat.from_vectors(pelvis_current[:3], pelvis_current[-4:]), TF_mat.from_vectors(body_rel_pos, body_rel_quat))

        cup_rel_tf = TF_mat.mul(body_tf.inverse(), TF_mat.from_vectors(cup_pos, [0, 0, 0, 1]))
        grasp_pos, grasp_quat = TF_mat.mul(cup_rel_tf, ee2grasp.inverse()).as_vectors()


        planning_scene.set_planning_joint_index(32, 40)

        for _ in range(20):
            r, q_goal = tracik_right.solve(np.array(grasp_pos), np.array(grasp_quat), q_current[-8:])
            if r:
                print(q_goal)
                if planning_scene.is_valid(q_goal):
                    break
                else:
                    print('panda_right is in collision')
                    r = False
            else:
                print('panda_right IK failed')

        if r:
            rrt_planner = SuhanRRTConnect(state_dim=8, lb=joint_lower_limit, ub=joint_upper_limit, validity_fn=planning_scene.is_valid)
            rrt_planner.max_distance = 0.25
            rrt_planner.set_start(q_current[-8:])
            rrt_planner.set_goal(q_goal)
            for _ in range(10):
                r, path = rrt_planner.solve()
                if r:
                    traj_msg = traj_gen.ocp_trajectory(path, joint_upper_limit, joint_lower_limit, v_max)
                    save_traj(f2, traj_msg)
                    trajectory_pub.publish(traj_msg.joint_trajectory)
                    # prev_time = 0
                    # for point in traj_msg.joint_trajectory.points:
                    #     point_wb = trajectory_msgs.msg.JointTrajectoryPoint()
                    #     point_wb.positions = [*q_current[:-8], *point.positions]
                    #     point_wb.velocities = [*np.zeros(25), *point.velocities]
                    #     trajectory_pub.publish(point_wb)
                    #     print(point_wb)
                    #     cur_time = point.time_from_start.to_sec()
                    #     time.sleep(cur_time - prev_time)
                    #     prev_time = cur_time
                    break
    