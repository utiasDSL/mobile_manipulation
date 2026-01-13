"""Unified planners that can handle base and/or end-effector references."""

import logging
from abc import ABC, abstractmethod

import numpy as np

from mm_utils.enums import RefType
from mm_utils.math import interpolate, wrap_pi_array, wrap_pi_scalar


class Planner(ABC):
    """Base class for planners that can provide base and/or EE references."""

    def __init__(self, name, ref_type):
        self.py_logger = logging.getLogger("Planner")
        self.name = name
        self.ref_type = ref_type
        self.robot_states = None
        self.close_to_finish = False
        self.started = False

        # Track which references this planner provides
        self.has_base_ref = False
        self.has_ee_ref = False

    def getBaseTrackingPoint(self, t, robot_states=None):
        """Get base tracking point (position and velocity).

        Returns None, None if this planner doesn't provide base references.

        Args:
            t (float): Time (s).
            robot_states (dict, optional): (joint angle, joint velocity), optional.

        Returns:
            (position, velocity) or (None, None) if not applicable
        """
        return None, None

    def getEETrackingPoint(self, t, robot_states=None):
        """Get EE tracking point (position and velocity).

        Returns None, None if this planner doesn't provide EE references.

        Args:
            t (float): Time (s).
            robot_states (dict, optional): (joint angle, joint velocity), optional.

        Returns:
            (position, velocity) or (None, None) if not applicable
        """
        return None, None

    def getBaseTrackingPointArray(self, robot_states, num_pts, dt, time_offset=0):
        """Get array of base tracking points along the path.

        Returns None, None if this planner doesn't provide base references.

        Args:
            robot_states (dict): Current robot states.
            num_pts (int): Number of points to return.
            dt (float): Time step between points.
            time_offset (float): Time offset from current time.

        Returns:
            (positions, velocities) arrays or (None, None) if not applicable
        """
        return None, None

    def getEETrackingPointArray(self, robot_states, num_pts, dt, time_offset=0):
        """Get array of EE tracking points along the path.

        Returns None, None if this planner doesn't provide EE references.

        Args:
            robot_states (dict): Current robot states.
            num_pts (int): Number of points to return.
            dt (float): Time step between points.
            time_offset (float): Time offset from current time.

        Returns:
            (positions, velocities) arrays or (None, None) if not applicable
        """
        return None, None

    @abstractmethod
    def checkFinished(self, t, states):
        """Check if the planner has finished.

        Args:
            t (float): Current time.
            states (dict): Dictionary with "base" and "EE" states:
                - "base": {"pose": [x, y, yaw], "velocity": [vx, vy, vyaw]}
                - "EE": {"pose": [x, y, z, roll, pitch, yaw], "velocity": [vx, vy, vz, wx, wy, wz]}

        Returns:
            bool: True if finished, False otherwise
        """
        return False

    def updateRobotStates(self, robot_states):
        """Update robot states.

        Args:
            robot_states (tuple): (joint angle, joint velocity).
        """
        self.robot_states = robot_states

    def ready(self):
        """Check if planner is ready.

        Returns:
            bool: True if ready
        """
        return True

    def closeToFinish(self):
        """Check if planner is close to finish.

        Returns:
            bool: True if close to finish
        """
        return self.close_to_finish

    def activate(self):
        """Activate the planner."""
        self.started = True


class WaypointPlanner(Planner):
    """Unified waypoint planner supporting optional base and/or EE targets.

    Can specify:
    - base_pose only: Move base to target
    - ee_pose only: Move EE to target
    - both: Move base and EE simultaneously
    """

    def __init__(self, config):
        super().__init__(name=config["name"], ref_type=RefType.WAYPOINT)

        # Parse base pose (optional)
        if "base_pose" in config:
            self.base_target = np.array(config["base_pose"])
            if len(self.base_target) != 3:
                raise ValueError(
                    f"base_pose must be SE2 [x, y, yaw], got {len(self.base_target)} dimensions"
                )
            self.has_base_ref = True
        else:
            self.base_target = None
            self.has_base_ref = False

        # Parse EE pose (optional)
        if "ee_pose" in config:
            self.ee_target = np.array(config["ee_pose"])
            if len(self.ee_target) != 6:
                raise ValueError(
                    f"ee_pose must be SE3 [x, y, z, roll, pitch, yaw], got {len(self.ee_target)} dimensions"
                )
            self.has_ee_ref = True
        else:
            self.ee_target = None
            self.has_ee_ref = False

        if not self.has_base_ref and not self.has_ee_ref:
            raise ValueError(
                "WaypointPlanner must specify at least one of 'base_pose' or 'ee_pose'"
            )

        # Common parameters
        self.tracking_err_tol = config.get("tracking_err_tol", 0.02)
        self.tracking_ori_err_tol = config.get("tracking_ori_err_tol", 0.1)
        self.hold_period = config.get("hold_period", 0.0)

        # State tracking
        self.finished = False
        self.base_reached = False
        self.ee_reached = False
        self.t_reached = 0

    def getBaseTrackingPoint(self, t, robot_states=None):
        """Get base tracking point.

        Args:
            t (float): Current time.
            robot_states (dict, optional): Current robot states (unused for waypoint planner).

        Returns:
            tuple: (position, velocity) where position is (3,) array and velocity is (3,) array, or (None, None) if no base reference.
        """
        if not self.has_base_ref:
            return None, None
        return self.base_target.copy(), np.zeros(3)

    def getEETrackingPoint(self, t, robot_states=None):
        """Get EE tracking point.

        Args:
            t (float): Current time.
            robot_states (dict, optional): Current robot states (unused for waypoint planner).

        Returns:
            tuple: (position, velocity) where position is (6,) array [x,y,z,qx,qy,qz,qw] and velocity is (6,) array, or (None, None) if no EE reference.
        """
        if not self.has_ee_ref:
            return None, None
        return self.ee_target.copy(), np.zeros(6)

    def checkFinished(self, t, states):
        """Check if waypoint has been reached.

        Args:
            t (float): Current time.
            states (dict): Dictionary with "base" and "EE" keys containing pose information.

        Returns:
            bool: True if waypoint has been reached, False otherwise.
        """
        base_finished = True
        ee_finished = True

        # Check base if applicable
        if self.has_base_ref:
            base_pose = states["base"]["pose"]
            pos_err = np.linalg.norm(base_pose[:2] - self.base_target[:2])
            yaw_err = abs(wrap_pi_scalar(base_pose[2] - self.base_target[2]))
            pos_within_tol = pos_err < self.tracking_err_tol
            ori_within_tol = yaw_err < self.tracking_ori_err_tol

            if pos_within_tol and ori_within_tol:
                if not self.base_reached:
                    self.base_reached = True
                    self.t_reached = t
                    self.py_logger.info(
                        f"{self.name} base reached (pos_err: {pos_err:.4f}, ori_err: {yaw_err:.4f})"
                    )
                base_finished = True
            else:
                base_finished = False
                if self.base_reached:
                    self.base_reached = False
                    self.t_reached = 0

        # Check EE if applicable
        if self.has_ee_ref:
            ee_pose = states["EE"]["pose"]
            pos_err = np.linalg.norm(ee_pose[:3] - self.ee_target[:3])
            ori_diff = wrap_pi_array(ee_pose[3:] - self.ee_target[3:])
            ori_err = np.linalg.norm(ori_diff)
            pos_within_tol = pos_err < self.tracking_err_tol
            ori_within_tol = ori_err < self.tracking_ori_err_tol

            if pos_within_tol and ori_within_tol:
                if not self.ee_reached:
                    self.ee_reached = True
                    if not self.base_reached:  # Only set time if base hasn't set it
                        self.t_reached = t
                    self.py_logger.info(
                        f"{self.name} EE reached (pos_err: {pos_err:.4f}, ori_err: {ori_err:.4f})"
                    )
                ee_finished = True
            else:
                ee_finished = False
                if self.ee_reached:
                    self.ee_reached = False
                    if not self.base_reached:
                        self.t_reached = 0

        # Check hold period
        if (self.base_reached or self.ee_reached) and self.hold_period > 0:
            if (t - self.t_reached) < self.hold_period:
                self.finished = False
                return False

        # Finished if all specified targets are reached
        self.finished = base_finished and ee_finished
        if self.finished and not (self.base_reached and self.ee_reached):
            # At least one target reached, mark as finished
            self.py_logger.info(f"{self.name} finished")

        return self.finished

    def reset(self):
        """Reset planner state."""
        self.finished = False
        self.base_reached = False
        self.ee_reached = False
        self.t_reached = 0
        self.py_logger.info(f"{self.name} reset")


class PathPlanner(Planner):
    """Unified path planner supporting optional base and/or EE paths.

    Can specify:
    - base_path only: Follow base path
    - ee_path only: Follow EE path
    - both: Follow base and EE paths simultaneously
    """

    def __init__(self, config):
        super().__init__(name=config["name"], ref_type=RefType.PATH)

        # Parse base path (optional)
        if "base_path" in config:
            base_path = np.array(config["base_path"])
            if base_path.shape[1] != 3:
                raise ValueError(
                    f"base_path must be SE2 [x, y, yaw], got shape {base_path.shape}"
                )
            self.has_base_ref = True
            self.base_plan = self._create_plan(base_path, config.get("dt", 0.01))
        else:
            self.base_plan = None
            self.has_base_ref = False

        # Parse EE path (optional)
        if "ee_path" in config:
            ee_path = np.array(config["ee_path"])
            if ee_path.shape[1] != 6:
                raise ValueError(
                    f"ee_path must be SE3 [x, y, z, roll, pitch, yaw], got shape {ee_path.shape}"
                )
            self.has_ee_ref = True
            self.ee_plan = self._create_plan(ee_path, config.get("dt", 0.01))
        else:
            self.ee_plan = None
            self.has_ee_ref = False

        if not self.has_base_ref and not self.has_ee_ref:
            raise ValueError(
                "PathPlanner must specify at least one of 'base_path' or 'ee_path'"
            )

        # Common parameters
        self.tracking_err_tol = config.get("tracking_err_tol", 0.02)
        self.tracking_ori_err_tol = config.get("tracking_ori_err_tol", 0.1)
        self.end_stop = config.get("end_stop", False)

        # State tracking
        self.finished = False
        self.started = False
        self.start_time = 0

    def _create_plan(self, path, dt):
        """Create plan dictionary with times, positions, and velocities."""
        velocities = np.zeros_like(path)
        if len(path) > 1:
            velocities[:-1] = np.diff(path, axis=0) / dt
            velocities[-1] = velocities[-2]  # Extend last velocity

        times = np.arange(len(path)) * dt

        return {
            "t": times,
            "p": path,
            "v": velocities,
        }

    def getBaseTrackingPoint(self, t, robot_states=None):
        """Get base tracking point from path.

        Args:
            t (float): Current time.
            robot_states (dict, optional): Current robot states (unused for path planner).

        Returns:
            tuple: (position, velocity) where position is (3,) array and velocity is (3,) array, or (None, None) if no base reference.
        """
        if not self.has_base_ref:
            return None, None

        if not self.started:
            self.started = True
            self.start_time = t

        te = t - self.start_time
        p, v = interpolate(te, self.base_plan)
        return p, v

    def getEETrackingPoint(self, t, robot_states=None):
        """Get EE tracking point from path.

        Args:
            t (float): Current time.
            robot_states (dict, optional): Current robot states (unused for path planner).

        Returns:
            tuple: (position, velocity) where position is (6,) array [x,y,z,qx,qy,qz,qw] and velocity is (6,) array, or (None, None) if no EE reference.
        """
        if not self.has_ee_ref:
            return None, None

        if not self.started:
            self.started = True
            self.start_time = t

        te = t - self.start_time
        p, v = interpolate(te, self.ee_plan)
        return p, v

    def getBaseTrackingPointArray(self, robot_states, num_pts, dt, time_offset=0):
        """Get array of base tracking points.

        Args:
            robot_states (dict): Current robot states (unused, kept for interface compatibility).
            num_pts (int): Number of points to return.
            dt (float): Time step between points.
            time_offset (float): Time offset from path start (elapsed time since path started).
        """
        if (
            not self.has_base_ref
            or self.base_plan is None
            or len(self.base_plan["t"]) == 0
        ):
            return None, None

        # Times are relative to plan start (plan times start at 0)
        times = time_offset + np.arange(num_pts) * dt
        positions = np.array([interpolate(t, self.base_plan)[0] for t in times])
        velocities = np.array([interpolate(t, self.base_plan)[1] for t in times])
        return positions, velocities

    def getEETrackingPointArray(self, robot_states, num_pts, dt, time_offset=0):
        """Get array of EE tracking points.

        Args:
            robot_states (dict): Current robot states (unused, kept for interface compatibility).
            num_pts (int): Number of points to return.
            dt (float): Time step between points.
            time_offset (float): Time offset from path start (elapsed time since path started).
        """
        if not self.has_ee_ref or self.ee_plan is None or len(self.ee_plan["t"]) == 0:
            return None, None

        # Times are relative to plan start (plan times start at 0)
        times = time_offset + np.arange(num_pts) * dt
        positions = np.array([interpolate(t, self.ee_plan)[0] for t in times])
        velocities = np.array([interpolate(t, self.ee_plan)[1] for t in times])
        return positions, velocities

    def _compute_error(self, curr_pose, end_pose, is_base):
        """Compute position and orientation errors."""
        if is_base:
            pos_err = np.linalg.norm(curr_pose[:2] - end_pose[:2])
            yaw_err = abs(wrap_pi_scalar(curr_pose[2] - end_pose[2]))
            return pos_err, yaw_err
        else:  # EE
            pos_err = np.linalg.norm(curr_pose[:3] - end_pose[:3])
            ori_diff = wrap_pi_array(curr_pose[3:] - end_pose[3:])
            ori_err = np.linalg.norm(ori_diff)
            return pos_err, ori_err

    def checkFinished(self, t, states):
        """Check if path has been completed.

        Args:
            t (float): Current time.
            states (dict): Dictionary with "base" and "EE" keys containing pose information.

        Returns:
            bool: True if path has been completed, False otherwise.
        """
        base_finished = True
        ee_finished = True

        # Check base if applicable
        if self.has_base_ref:
            base_pose = states["base"]["pose"]
            base_vel = states["base"].get("velocity")
            end_pose = self.base_plan["p"][-1]
            pos_err, ori_err = self._compute_error(base_pose, end_pose, is_base=True)
            pos_cond = pos_err < self.tracking_err_tol
            ori_cond = ori_err < self.tracking_ori_err_tol
            pos_ori_cond = pos_cond and ori_cond
            vel_cond = base_vel is not None and np.linalg.norm(base_vel) < 1e-2

            if (not self.end_stop and pos_ori_cond) or (
                self.end_stop and pos_ori_cond and vel_cond
            ):
                base_finished = True
            else:
                base_finished = False

        # Check EE if applicable
        if self.has_ee_ref:
            ee_pose = states["EE"]["pose"]
            ee_vel = states["EE"].get("velocity")
            end_pose = self.ee_plan["p"][-1]
            pos_err, ori_err = self._compute_error(ee_pose, end_pose, is_base=False)
            pos_cond = pos_err < self.tracking_err_tol
            ori_cond = ori_err < self.tracking_ori_err_tol
            pos_ori_cond = pos_cond and ori_cond
            vel_cond = ee_vel is not None and np.linalg.norm(ee_vel) < 1e-2

            if (not self.end_stop and pos_ori_cond) or (
                self.end_stop and pos_ori_cond and vel_cond
            ):
                ee_finished = True
            else:
                ee_finished = False

        # Finished if all specified paths are completed
        self.finished = base_finished and ee_finished
        if self.finished:
            self.py_logger.info(f"{self.name} finished")

        return self.finished

    def reset(self):
        """Reset planner state."""
        self.finished = False
        self.started = False
        self.start_time = 0
        self.py_logger.info(f"{self.name} reset")


def create_planner(config: dict):
    """Create a planner instance from configuration.

    Args:
        config (dict): Planner configuration dictionary containing "planner_type".

    Returns:
        Instance of the requested planner

    Raises:
        ValueError: If the planner type is unknown or missing
    """
    if "planner_type" not in config:
        raise ValueError("Configuration missing 'planner_type' field")

    planner_type = config["planner_type"]

    if planner_type == "WaypointPlanner":
        return WaypointPlanner(config)
    elif planner_type == "PathPlanner":
        return PathPlanner(config)
    else:
        raise ValueError(
            f"Unknown planner type: '{planner_type}'. "
            f"Available: WaypointPlanner, PathPlanner"
        )
