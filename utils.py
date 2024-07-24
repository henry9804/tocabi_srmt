import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose

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

def vectors_to_pose(pos=None, ori=None):
    pose = Pose()
    pose.orientation.w = 1

    if pos != None:
        pose.position.x = pos[0]
        pose.position.y = pos[1]
        pose.position.z = pos[2]

    if ori != None:
        if len(ori) == 3:
            quat = Rotation.from_euler('xyz', ori).as_quat()
        else:
            quat = ori
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

    return pose
