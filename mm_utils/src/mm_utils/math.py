import casadi as cs
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from spatialmath.base import q2r, qunit, r2q

QUAT_ORDER = "xyzs"


def quat_to_rot(q):
    """Convert quaternion to rotation matrix.

    Args:
        q (ndarray): Quaternion, shape (4,).

    Returns:
        ndarray: 3x3 rotation matrix.
    """
    return q2r(q, order=QUAT_ORDER)


def rot_to_quat(C):
    """Convert rotation matrix to quaternion.

    Args:
        C (ndarray): 3x3 rotation matrix.

    Returns:
        ndarray: Quaternion, shape (4,).
    """
    return r2q(C, order=QUAT_ORDER)


def quat_multiply(q0, q1, normalize=True):
    """Hamilton product of two quaternions.

    Args:
        q0 (ndarray): First quaternion, shape (4,).
        q1 (ndarray): Second quaternion, shape (4,).
        normalize (bool): If True, normalize quaternions before multiplication.

    Returns:
        ndarray: Product quaternion, shape (4,).
    """
    if normalize:
        q0 = qunit(q0)
        q1 = qunit(q1)
    C0 = quat_to_rot(q0)
    C1 = quat_to_rot(q1)
    return rot_to_quat(C0 @ C1)


def quat_rotate(q, r):
    """Rotate point by rotation represented by quaternion.

    Args:
        q (ndarray): Quaternion representing rotation, shape (4,).
        r (ndarray): Point to rotate, shape (3,).

    Returns:
        ndarray: Rotated point, shape (3,).
    """
    return quat_to_rot(q) @ r


def quat_transform(r_ba_a, q_ab, r_cb_b):
    """Transform point by rotating and translating.

    Args:
        r_ba_a (ndarray): Translation vector from frame A to frame B in frame A, shape (3,).
        q_ab (ndarray): Quaternion representing rotation from frame A to frame B, shape (4,).
        r_cb_b (ndarray): Point in frame B, shape (3,).

    Returns:
        ndarray: Transformed point in frame A, shape (3,).
    """
    return quat_rotate(q_ab, r_cb_b) + r_ba_a


def quat_inverse(q):
    """Compute inverse of quaternion.

    Such that quat_multiply(q, quat_inverse(q)) = [0, 0, 0, 1].

    Args:
        q (ndarray): Quaternion, shape (4,).

    Returns:
        ndarray: Inverse quaternion, shape (4,).
    """
    return np.append(-q[:3], q[3])


def make_trans_from_vec(rotvec, pos):
    """Create 4x4 transformation matrix from rotation vector and position.

    Args:
        rotvec (ndarray): Rotation vector (axis-angle representation).
        pos (ndarray): Translation vector, shape (3,).

    Returns:
        ndarray: 4x4 homogeneous transformation matrix.
    """
    R = Rot.from_rotvec(rotvec).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def wrap_pi_scalar(theta):
    """Wrap angle to [-pi, pi] range.

    Args:
        theta (float): Angle in radians.

    Returns:
        float: Wrapped angle in [-pi, pi].
    """
    while np.abs(theta) > np.pi:
        if theta > np.pi:
            theta -= 2 * np.pi
        elif theta < -np.pi:
            theta += 2 * np.pi

    return theta


def wrap_pi_array(thetas):
    """Wrap array of angles to [-pi, pi] range.

    Args:
        thetas (ndarray): Array of angles in radians.

    Returns:
        ndarray: Array of wrapped angles in [-pi, pi].
    """
    thetas_wrapped = [wrap_pi_scalar(theta) for theta in thetas]

    return np.array(thetas_wrapped)


def rms_continuous(ts, data):
    """RMS of data over a period of time.

    Args:
        ts (ndarray): 1D array of length N, time stamp of each row of data.
        data (ndarray): 2D array, N x data dimension.

    Returns:
        ndarray: RMS values over time period.
    """
    dts = ts[1:] - ts[:-1]
    dts = np.hstack((dts, dts[-1]))
    rms = (np.sum(data**2 * dts, axis=0) / (ts[-1] - ts[0])) ** 0.5

    return rms


def integrate_zoh(ts, data):
    """Numerical integration (ZOH) of data over a period of time.

    Args:
        ts (ndarray): 1D array of length N, time stamp of each row of data.
        data (ndarray): 2D array, N x data dimension.

    Returns:
        ndarray: Integrated values over time period.
    """
    dts = ts[1:] - ts[:-1]
    dts = np.hstack((dts, dts[-1]))
    integral = np.sum(data * dts, axis=0)

    return integral / (ts[-1] - ts[0])


def statistics(data):
    """Compute mean, max, and min statistics along the first axis.

    Args:
        data (ndarray): Input data array.

    Returns:
        tuple: (mean, max, min) arrays.
    """
    mean = np.mean(data, axis=0)
    max = np.max(data, axis=0)
    min = np.min(data, axis=0)

    return mean, max, min


def statistics_std(data):
    """Compute standard deviation along the first axis.

    Args:
        data (ndarray): Input data array.

    Returns:
        ndarray: Standard deviation array.
    """
    std = np.std(data, axis=0)

    return std


def normalize_wrt_bounds(lower_bound, upper_bound, data):
    """Normalize data with respect to bounds.
        # -1 --> saturate lower bounds
        # 1  --> saturate upper bounds
        # 0  --> mean

    Args:
        lower_bound (ndarray): 1D array same length of data (n).
        upper_bound (ndarray): 1D array same length of data (n).
        data (ndarray): 2D array time dim (N) x data dim (n).

    Returns:
        ndarray: Normalized data.
    """

    mean_bound = (upper_bound + lower_bound) / 2
    bound_width = upper_bound - mean_bound
    data_normalized = (data - mean_bound) / bound_width

    return data_normalized


def interpolate(t, plan):
    """Interpolate position and velocity from a trajectory plan at time t.

    Args:
        t (float): Time to interpolate at.
        plan (dict): Dictionary with keys 't' (time array), 'p' (position array), 'v' (velocity array).

    Returns:
        tuple: (interpolated_position, interpolated_velocity).
    """
    if t >= plan["t"][-1]:
        return plan["p"][-1], np.zeros_like(plan["p"][-1])
    elif t <= plan["t"][0]:
        return plan["p"][0], np.zeros_like(plan["p"][0])

    indx = np.argwhere(plan["t"] < t)[-1][0]
    dt = plan["t"][indx + 1] - plan["t"][indx]

    p0 = plan["p"][indx]
    p1 = plan["p"][indx + 1]
    p = (p1 - p0) / dt * (t - plan["t"][indx]) + p0

    v0 = plan["v"][indx]
    v1 = plan["v"][indx + 1]
    v = (v1 - v0) / dt * (t - plan["t"][indx]) + v0

    return p, v


def casadi_SO2(theta):
    """Create SO(2) rotation matrix from angle using CasADi.

    Args:
        theta (casadi.MX): Rotation angle in radians.

    Returns:
        casadi.MX: 2x2 rotation matrix.
    """
    R = cs.MX(2, 2)
    R[0, 0] = cs.cos(theta)
    R[1, 1] = cs.cos(theta)
    R[1, 0] = cs.sin(theta)
    R[0, 1] = -cs.sin(theta)

    return R


def casadi_SO3_Rx(theta):
    """Create SO(3) rotation matrix about x-axis from angle using CasADi.

    Args:
        theta (casadi.MX): Rotation angle in radians.

    Returns:
        casadi.MX: 3x3 rotation matrix about x-axis.
    """
    R = cs.MX(3, 3)
    R[0, 0] = cs.cos(theta)
    R[1, 1] = cs.cos(theta)
    R[1, 0] = cs.sin(theta)
    R[0, 1] = -cs.sin(theta)
    R[2, 2] = 1

    return R


def casadi_SO3_log(R):
    """Compute logarithm map of SO(3) rotation matrix using CasADi.

    Args:
        R (casadi.MX): 3x3 rotation matrix.

    Returns:
        casadi.MX: Rotation vector (axis-angle representation), shape (3,).
    """
    theta = cs.acos((cs.trace(R) - 1) / 2)
    coeff_large_angle = theta / (2 * cs.sin(theta))
    coeff_small_angle = theta / (2 * (theta - theta**3 / 6 + theta**5 / 120))
    omega_cross_large_angle = coeff_large_angle * (R - R.T)
    omega_cross_small_angle = coeff_small_angle * (R - R.T)
    omega_large_angle = cs.vertcat(
        omega_cross_large_angle[2, 1],
        omega_cross_large_angle[0, 2],
        omega_cross_large_angle[1, 0],
    )
    omega_small_angle = cs.vertcat(
        omega_cross_small_angle[2, 1],
        omega_cross_small_angle[0, 2],
        omega_cross_small_angle[1, 0],
    )

    omega_list = [omega_small_angle, omega_large_angle]
    omega = cs.conditional(theta > 1e-2, omega_list, 0, False)

    return omega
