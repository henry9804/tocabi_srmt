import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose

class TF_mat:
    def __init__(self, T=None):
        if T is None:
            self.T = np.identity(4)
        else:
            self.T = T

    @classmethod
    def from_vectors(cls, pos, quat):
        tf_mat = cls()
        tf_mat.T[:3,:3] = Rotation.from_quat(quat).as_matrix()
        tf_mat.T[:3,3] = pos

        return tf_mat
    
    @classmethod
    def from_pose_msg(cls, pose):
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

        tf_mat = cls()
        tf_mat.T[:3,:3] = Rotation.from_quat(quat).as_matrix()
        tf_mat.T[:3,3] = pos

        return tf_mat
    
    @classmethod
    def mul(cls, tf1, tf2):
        tf_mat = cls(np.matmul(tf1.T, tf2.T))

        return tf_mat
    
    def inverse(self):
        p = self.T[:3,3]
        R = self.T[:3,:3]
        inv = np.identity(4)
        inv[:3,:3] = R.transpose()
        inv[:3,3] = -np.matmul(R.transpose(), p)

        return TF_mat(inv)
    
    def as_matrix(self):
        return self.T
    
    def as_vectors(self):
        p = self.T[:3,3]
        R = self.T[:3,:3]
        q = Rotation.from_matrix(R).as_quat()

        return p, q
    
    def as_pose_msg(self):
        p = self.T[:3,3]
        R = self.T[:3,:3]
        q = Rotation.from_matrix(R).as_quat()

        pose = Pose()
        pose.position.x = p[0]
        pose.position.y = p[1]
        pose.position.z = p[2]
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        return pose
