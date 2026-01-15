"""TaskManager - manages task execution and extracts references for MPC."""

import logging

import numpy as np

from mm_plan.Planners import create_planner
from mm_utils.enums import RefType


class TaskManager:
    """Task manager that executes tasks sequentially and extracts references for MPC."""

    def __init__(self, config):
        self.config = config
        self.started = False
        self.planners = [create_planner(task) for task in config["tasks"]]
        self.planner_num = len(self.planners)
        self.curr_task_id = 0
        self.logger = logging.getLogger("Planner")

    def activatePlanners(self):
        """Activate all planners."""
        for planner in self.planners:
            planner.activate()

    def getPlanner(self):
        """Get the active planner (current task).

        Returns:
            Active planner
        """
        return self.planners[self.curr_task_id]

    def getReferences(self, t, robot_states, num_horizon_points, dt):
        """Extract references from active planners for MPC.

        This method extracts base and EE references from all active planners
        and returns them in a format the MPC can use. If multiple planners
        provide references for the same type (base or EE), the first one takes
        priority (current task).

        Args:
            t (float): Current time.
            robot_states (tuple): (joint angles, joint velocities).
            num_horizon_points (int): Number of horizon points (N+1).
            dt (float): Time step.

        Returns:
            Dictionary with reference trajectories:
            {
                "base_pose": array of shape (N+1, 3) or None,
                "base_velocity": array of shape (N+1, 3) or None,
                "ee_pose": array of shape (N+1, 6) or None,
                "ee_velocity": array of shape (N+1, 6) or None,
            }
        """
        planner = self.getPlanner()

        # Initialize reference arrays
        base_pose_ref = None
        base_vel_ref = None
        ee_pose_ref = None
        ee_vel_ref = None

        planner.updateRobotStates(robot_states)

        # Initialize path planner start time if it's the current task and hasn't started
        is_current_task = (
            planner == self.planners[self.curr_task_id]
            if self.curr_task_id < self.planner_num
            else False
        )
        if is_current_task and planner.ref_type == RefType.PATH and not planner.started:
            planner.started = True
            planner.start_time = t

        # Process base references
        if planner.has_base_ref:
            if planner.ref_type == RefType.PATH:
                # For path planners, get array starting from current position in path
                # Calculate time elapsed since path started
                time_elapsed = t - planner.start_time if planner.started else 0
                p_array, v_array = planner.getBaseTrackingPointArray(
                    robot_states, num_horizon_points, dt, time_offset=time_elapsed
                )
                if p_array is not None:
                    base_pose_ref = p_array
                    base_vel_ref = v_array
            else:  # WAYPOINT
                # For waypoints, create constant arrays
                p, v = planner.getBaseTrackingPoint(t, robot_states)
                if p is not None:
                    base_pose_ref = np.tile(p, (num_horizon_points, 1))
                    base_vel_ref = np.tile(v, (num_horizon_points, 1))

        # Process EE references
        if planner.has_ee_ref:
            if planner.ref_type == RefType.PATH:
                # For path planners, get array starting from current position in path
                time_elapsed = t - planner.start_time if planner.started else 0
                p_array, v_array = planner.getEETrackingPointArray(
                    robot_states, num_horizon_points, dt, time_offset=time_elapsed
                )
                if p_array is not None:
                    ee_pose_ref = p_array
                    ee_vel_ref = v_array
            else:  # WAYPOINT
                # For waypoints, create constant arrays
                p, v = planner.getEETrackingPoint(t, robot_states)
                if p is not None:
                    ee_pose_ref = np.tile(p, (num_horizon_points, 1))
                    ee_vel_ref = np.tile(v, (num_horizon_points, 1))

        return {
            "base_pose": base_pose_ref,
            "base_velocity": base_vel_ref,
            "ee_pose": ee_pose_ref,
            "ee_velocity": ee_vel_ref,
        }

    def update(self, t, states):
        """Update task manager and check if current task is finished.

        Args:
            t (float): Current time.
            states (dict): Dictionary with "EE" and "base" states:
                - "base": {"pose": [x, y, yaw], "velocity": [vx, vy, vyaw]}
                - "EE": {"pose": [x, y, z, roll, pitch, yaw], "velocity": [vx, vy, vz, wx, wy, wz]}

        Returns:
            (finished, increment): Whether current task finished, and increment (always 1)
        """
        if self.curr_task_id >= self.planner_num:
            return True, 1

        planner = self.planners[self.curr_task_id]

        # Initialize path planner start time when it becomes active
        if planner.ref_type == RefType.PATH and not planner.started:
            planner.started = True
            planner.start_time = t

        finished = planner.checkFinished(t, states)

        if finished:
            if self.curr_task_id < self.planner_num - 1:
                self.curr_task_id += 1
                self.logger.info(
                    "Task finished, moving to task %d/%d",
                    self.curr_task_id + 1,
                    self.planner_num,
                )
            else:
                self.logger.info(
                    "All tasks completed (%d/%d)", self.planner_num, self.planner_num
                )
            return True, 1
        else:
            self.logger.info(
                "Working on task %d/%d",
                self.curr_task_id + 1,
                self.planner_num,
            )
            return False, 1

    def print(self):
        """Print task manager status."""
        print(
            f"TaskManager: {self.planner_num} tasks, currently on task {self.curr_task_id + 1}"
        )
