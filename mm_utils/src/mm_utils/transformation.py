import numpy as np
from scipy.spatial.transform import Rotation as Rot


def make_trans_from_vec(rotvec, pos):
    R = Rot.from_rotvec(rotvec).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def make_trans_from_quat(quat, pos):
    R = Rot.from_quat(quat).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T
