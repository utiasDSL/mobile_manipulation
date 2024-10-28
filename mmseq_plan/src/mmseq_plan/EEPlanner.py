#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg
# from liegroups import SE3, SO3
from spatialmath.base import rotz, rpy2r, q2r,trnorm, tr2rpy
from spatialmath import SE3

from mmseq_plan.PlanBaseClass import Planner, TrajectoryPlanner
from mmseq_plan.BasePlanner import ROSTrajectoryPlanner,ROSTrajectoryPlannerOnDemand

from mmseq_utils.transformation import *
from mmseq_utils.parsing import parse_number

class EESimplePlanner(Planner):
    def __init__(self, planner_params):
        super().__init__(name=planner_params["name"],
                         type="EE", 
                         ref_type="waypoint", 
                         ref_data_type="Vec3",
                         frame_id=planner_params["frame_id"])

        self.target_pos = np.array(planner_params["target_pos"])

        self.started = False
        self.finished = False
        self.reached_target = False
        self.t_reached_target = 0
        self.hold_period = planner_params["hold_period"]
        self.tracking_err_tol = planner_params["tracking_err_tol"]


    def getTrackingPoint(self, t, robot_states=None):
        return self.target_pos, np.zeros(3)
    
    def checkFinished(self, t, ee_states):
        ee_curr_pos = ee_states[0]
        if np.linalg.norm(ee_curr_pos - self.target_pos) > self.tracking_err_tol:
            if self.reached_target:
                self.reset()
            return self.finished

        if not self.reached_target:
            self.reached_target = True
            self.t_reached_target=t
            self.py_logger.info(self.name + " Planner Reached Target.")
            return self.finished

        if t - self.t_reached_target > self.hold_period:
            self.finished = True
            self.py_logger.info(self.name + " Planner Finished.")

        return self.finished

    def reset(self):
        self.reached_target = False
        self.t_reached_target = 0
        self.finished = False
        self.started = False
        self.py_logger.info(self.name + " Planner Reset.")

    @staticmethod
    def getDefaultParams():
        config = {}
        config["name"] = "EE Position"
        config["planner_type"] = "EESimplePlanner"
        config["frame_id"] = "EE"
        config["target_pos"] = [0, 0, 0]
        config["hold_period"] = 3.
        config["tracking_err_tol"] = 0.02

        return config

class EESimplePlannerBaseFrame(Planner):
    def __init__(self, planner_params):
        super().__init__(name=planner_params["name"],
                         type="EE", 
                         ref_type="waypoint", 
                         ref_data_type="Vec3",
                         frame_id=planner_params["frame_id"])
        self.target_pos = np.array(planner_params["target_pos"])

        self.started = False
        self.finished = False
        self.reached_target = False
        self.t_reached_target = 0
        self.hold_period = planner_params["hold_period"]
        self.tracking_err_tol = planner_params["tracking_err_tol"]


    def getTrackingPoint(self, t, robot_states=None):
        # q,_ = robot_states
        # target_pos = self.target_pos.copy()
        # target_pos[:2] -= q[:2]
        # Rwb = rotz(q[2])
        # target_pos= Rwb.T @ target_pos

        return self.target_pos, np.zeros(3)
    
    def checkFinished(self, t, ee_states):
        ee_curr_pos = ee_states[0]
        if np.linalg.norm(ee_curr_pos - self.target_pos) > self.tracking_err_tol:
            if self.reached_target:
                self.reset()
            return self.finished

        if not self.reached_target:
            self.reached_target = True
            self.t_reached_target=t
            self.py_logger.info(self.name + " Planner Reached Target.")
            return self.finished

        if t - self.t_reached_target > self.hold_period:
            self.finished = True
            self.py_logger.info(self.name + " Planner Finished.")

        return self.finished

    def reset(self):
        self.reached_target = False
        self.t_reached_target = 0
        self.finished = False
        self.started = False
        self.py_logger.info(self.name + " Planner Reset.")

    @staticmethod
    def getDefaultParams():
        config = {}
        config["name"] = "EE Position"
        config["planner_type"] = "EESimplePlanner"
        config["frame_id"] = "EE"
        config["target_pos"] = [0, 0, 0]
        config["hold_period"] = 3.
        config["tracking_err_tol"] = 0.02

        return config




class EEPosTrajectoryCircle(Planner):
    def __init__(self, config):
        super().__init__(name=config["name"],
                         type="EE", 
                         ref_type="trajectory", 
                         ref_data_type="Vec3",
                         frame_id=config["frame_id"])
        self.tracking_err_tol = config["tracking_err_tol"]

        self.finished = False
        self.started = False
        self.start_time = 0

        self.T = config["period"]
        self.omega = np.pi * 2 / self.T
        self.c = np.array(config["center"])
        self.r = config["radius"]
        self.phi = parse_number(config["angular_offset"])
        self.round = int(config["round"])
        self.plane_id = config["plane_id"]

        self.dt = 0.01
        self.N = int(self.T * self.round / self.dt)
        self.plan = self._generatePlan()


    def _generatePlan(self):
        ts = np.linspace(0, self.T * self.round, self.N)
        pt1 = self.r * np.cos(self.omega * ts + self.phi)
        pt2 = self.r * np.sin(self.omega * ts + self.phi)
        pt = [pt1, pt2]
        plan_pos = np.repeat([self.c], self.N, axis=0)

        for i, pid in enumerate(self.plane_id):
            plan_pos[:, pid] += pt[i]

        v1 = -self.r * np.sin(self.omega * ts + self.phi) * self.omega
        v2 = self.r * np.cos(self.omega * ts + self.phi) * self.omega
        v = np.array([v1, v2])
        plan_vel = np.zeros_like(plan_pos)

        for i, pid in enumerate(self.plane_id):
            plan_vel[:, pid] += v[i]

        return {"p": plan_pos, "v": plan_vel}

    def _interpolate(self, t, plan, dt):
        plan_len = len(plan["p"])
        indx = int(t / dt)
        if indx > plan_len-2:
            return plan['p'][-1], np.zeros_like(plan['p'][-1])
        elif indx < 0:
            return plan['p'][0], np.zeros_like(plan['p'][0])

        p0 = plan["p"][indx]
        p1 = plan["p"][indx + 1]
        p = (p1 - p0) / dt * (t - indx * dt) + p0

        v0 = plan["v"][indx]
        v1 = plan["v"][indx + 1]
        v = (v1 - v0) / dt * (t - indx * dt) + v0

        return p, v

    def getTrackingPoint(self, t, robot_states=None):

        if self.started and self.start_time == 0:
            self.start_time = t

        te = t - self.start_time

        p, v = self._interpolate(te, self.plan, self.dt)

        return p, v

    def checkFinished(self, t, ee_states):
        ee_curr_pos = ee_states[0]
        if t - self.start_time > self.T * (self.round - 1):
            if np.linalg.norm(ee_curr_pos - self.plan[0]) < self.tracking_err_tol:
                self.finished = True
                self.py_logger.info(self.name + " Planner Finished")
        return self.finished

    def reset(self):
        self.finished = False


class EEPosTrajectoryLine(TrajectoryPlanner):
    def __init__(self, config):
        super().__init__(name=config["name"],
                         type="EE", 
                         ref_type="trajectory", 
                         ref_data_type="Vec3",
                         frame_id=config["frame_id"])
        self.tracking_err_tol = config["tracking_err_tol"]
        self.end_stop = config.get("end_stop", False)

        self.finished = False
        self.started = False
        self.start_time = 0
        self.reached_target = False

        self.initial_pos = np.array(config["initial_pos"])
        self.target_pos = np.array(config["target_pos"])
        self.cruise_speed = config["cruise_speed"]

        self.dt = 0.01
        self.plan = self._generatePlan()


    def regeneratePlan(self):
        self.plan = self._generatePlan()
        self.start_time = 0

    def _generatePlan(self):
        self.T = np.linalg.norm(self.initial_pos - self.target_pos) / self.cruise_speed

        ts = np.linspace(0, self.T, int(self.T/self.dt)).reshape((-1, 1))
        n = (self.target_pos - self.initial_pos) / np.linalg.norm(self.initial_pos - self.target_pos)
        plan_pos = n * ts * self.cruise_speed + self.initial_pos

        plan_vel = np.tile(n * self.cruise_speed, (int(self.T/self.dt), 1))

        return {'t': ts, 'p': plan_pos, 'v': plan_vel}

    def getTrackingPoint(self, t, robot_states=None):

        if self.started and self.start_time == 0:
            self.start_time = t

        te = t - self.start_time

        p, v = self._interpolate(te, self.plan)

        return p, v

    def checkFinished(self, t, ee_states):
        ee_curr_pos = ee_states[0]
        if np.linalg.norm(ee_curr_pos - self.target_pos) < self.tracking_err_tol:
            return True
        else:
            return False

    def reset(self):
        self.reached_target = False
        self.finished = False
        self.start_time = 0
        self.py_logger.info(self.name + " planner reset.")

    @staticmethod
    def getDefaultParams():
        config = {}
        config["name"] = "Base Position"
        config["planner_type"] = "BasePosTrajectoryLine"
        config["frame_id"] = "base"
        config["initial_pos"] = [0, 0]
        config["target_pos"] = [0, 0]
        config["cruise_speed"] = 0.5
        config["tracking_err_tol"] = 0.02

        return config

class EEPoseSE3Waypoint(Planner):
    def __init__(self, config):
        super().__init__(name=config["name"],
                         type="EE", 
                         ref_type="waypoint", 
                         ref_data_type="SE3",
                         frame_id=config["frame_id"])
        self.finished = False
        self.reached_target = False
        self.stamp = 0
        self.hold_period = config["hold_period"]
        self.target_pose = np.array(config["target_pose"])
        self.tracking_err_tol = config["tracking_err_tol"]

        
    def getTrackingPoint(self, t, robot_states=None):

        return self.target_pose, np.zeros(6)
    
    def checkFinished(self, t, ee_states):
        ee_curr_pos = ee_states[0]
        self.target_pos = self.target_pose[:3]
        if np.linalg.norm(ee_curr_pos - self.target_pos) > self.tracking_err_tol:
            if self.reached_target:
                self.reset()
            return self.finished

        if not self.reached_target:
            self.reached_target = True
            self.t_reached_target=t
            self.py_logger.info(self.name + " Planner Reached Target.")
            return self.finished

        if t - self.t_reached_target > self.hold_period:
            self.finished = True
            self.py_logger.info(self.name + " Planner Finished.")

        return self.finished
    
    def reset(self):
        self.reached_target = False
        self.stamp = 0
        self.finished = False


class EEPos3WaypointOnDemand(EESimplePlanner):

    def __init__(self, config):
        self.target_waypoints = np.array(config["target_waypoints"])
        self.num_waypoint = len(self.target_waypoints)
        self.curr_waypoint_idx = self.num_waypoint-1

        config["target_pos"] = self.target_waypoints[0]
        super().__init__(config)
    
    def regeneratePlan(self, states):
        self.curr_waypoint_idx += 1
        self.curr_waypoint_idx %= self.num_waypoint

        self.target_pos = self.target_waypoints[self.curr_waypoint_idx]
        self.reset()

class EEPoseSE3WaypointOnDemand(EEPoseSE3Waypoint):

    def __init__(self, config):
        self.target_waypoints = np.array(config["target_waypoints"])
        self.num_waypoint = len(self.target_waypoints)
        self.curr_waypoint_idx = self.num_waypoint-1

        config["target_pose"] = self.target_waypoints[0]
        super().__init__(config)
    
    def regeneratePlan(self, states):
        self.curr_waypoint_idx += 1
        self.curr_waypoint_idx %= self.num_waypoint

        self.target_pose = self.target_waypoints[self.curr_waypoint_idx]
        self.reset()

class EELookAhead(Planner):
    def __init__(self, config):
        super().__init__(name=config["name"],
                         type="EE", 
                         ref_type="path", 
                         ref_data_type="SE3",
                         frame_id=config["frame_id"])
        self.default_pose = np.array(config["default_pose"])
        self.default_rot = rpy2r(self.default_pose[3:])
        self.look_ahead_time = config["look_ahead_time"]
        self.max_angle = config["max_angle"]
        self.base_planner = None

    def set_base_planner(self, base_planner: ROSTrajectoryPlannerOnDemand):
        self.base_planner = base_planner

    def getTrackingPointArray(self, robot_states, num_pts, dt):
        # Get Base Ref Path Current
        curr_base_poses, _ = self.base_planner.getTrackingPointArray(robot_states, num_pts, dt)
        # Get Base Ref Path Future
        future_base_poses, _ = self.base_planner.getTrackingPointArray(robot_states, num_pts, dt, self.look_ahead_time)
        # Compute Angle Difference and new EE rpy
        new_poses = []

        for i in range(num_pts):
            r_bf_bc_w = future_base_poses[i, :2] - curr_base_poses[i, :2]

            Rw_bc = rotz(curr_base_poses[i, 2])
            r_bf_bc_bc = Rw_bc.T @ np.hstack((r_bf_bc_w,1))
            r_bf_eec_bc = r_bf_bc_bc[:2] - self.default_pose[:2]
            phi = np.arctan2(r_bf_eec_bc[1], r_bf_eec_bc[0])
            phi = min(phi, self.max_angle)
            phi = max(phi, -self.max_angle)
            new_rot = rotz(phi) @ self.default_rot
            new_rpy = tr2rpy(new_rot)

            new_poses.append(np.hstack((self.default_pose[:3], new_rpy)))

            # self.py_logger.debug("Future {}, Current, {}, phi {}, pose {}".format(future_base_poses[i, :2],
            #                                                                       curr_base_poses[i, :2],
            #                                                                       phi,
            #                                                                       new_poses[-1]))
        return np.array(new_poses), np.zeros(6)
    
    def getTrackingPoint(self, t, robot_states=None):
        poses, _ = self.getTrackingPointArray(robot_states, 2, 0.1)
        return poses[0], np.zeros(6)
    
    def checkFinished(self, t, ee_states):

        return False
    
    def reset(self):
        pass


class EELookAheadWorld(Planner):
    def __init__(self, config):
        super().__init__(name=config["name"],
                         type="EE", 
                         ref_type="path", 
                         ref_data_type="SE3",
                         frame_id=config["frame_id"])
        self.default_pose = np.array(config["default_pose"])
        self.default_rot = rpy2r(self.default_pose[3:])
        self.look_ahead_time = config["look_ahead_time"]
        self.max_angle = config["max_angle"]
        self.base_planner = None

    def set_base_planner(self, base_planner: ROSTrajectoryPlannerOnDemand):
        self.base_planner = base_planner

    def getTrackingPointArray(self, robot_states, num_pts, dt):
        # Get Base Ref Path Current
        curr_base_poses, _ = self.base_planner.getTrackingPointArray(robot_states, num_pts, dt)
        # Get Base Ref Path Future
        future_base_poses, _ = self.base_planner.getTrackingPointArray(robot_states, num_pts, dt, self.look_ahead_time)
        # Compute Angle Difference and new EE rpy
        new_poses = []

        for i in range(num_pts):
            r_bf_bc_w = future_base_poses[i, :2] - curr_base_poses[i, :2]

            Rw_bc = rotz(curr_base_poses[i, 2])
            r_bf_bc_bc = Rw_bc.T @ np.hstack((r_bf_bc_w,1))

            r_bf_eec_bc = r_bf_bc_bc[:2] - self.default_pose[:2]
            phi = np.arctan2(r_bf_bc_bc[1], r_bf_bc_bc[0])
            phi = min(phi, self.max_angle)
            phi = max(phi, -self.max_angle)

            Rbc_eec_new = rotz(phi) @ self.default_rot

            Rw_eec_new = Rw_bc @ Rbc_eec_new
            r_eec_w_new = np.hstack((curr_base_poses[i, :2], 0)) + Rw_bc @ self.default_pose[:3]
            new_rpy = tr2rpy(Rw_eec_new)

            new_poses.append(np.hstack((r_eec_w_new, new_rpy)))

            # self.py_logger.debug("Future {}, Current, {}, phi {}, pose {}".format(future_base_poses[i, :2],
            #                                                                       curr_base_poses[i, :2],
            #                                                                       phi,
            #                                                                       new_poses[-1]))
        return np.array(new_poses), np.zeros(6)
    
    def getTrackingPoint(self, t, robot_states=None):
        poses, _ = self.getTrackingPointArray(robot_states, 2, 0.1)
        return poses[0], np.zeros(6)
    
    def checkFinished(self, t, ee_states):

        return False
    
    def reset(self):
        pass


if __name__ == '__main__':
    planner_params = {"target_pose": [0., 1.,], 'hold_period':1}
    
    planner = EESimplePlanner(planner_params)
    planner_params = {"target_pose": [0., 1., 1.], 'hold_period':1}
    
    # T = make_trans_from_vec(np.array([0,0,1]) * np.pi/2, [1,0,0])
    # planner_params = {"target_pose": T, 'hold_period': 0}
    # planner = EESixDofWaypoint(planner_params)
    
    state_ee = make_trans_from_vec(np.array([0,0,1]) * np.pi/2*0.9, [1.,0,0])
    t = 0
    planner.checkFinished(t, state_ee)
    # sigma = 0.5
    # for i in range(6):
    #     sigma *= 0.5
    #     planner.regenerate(sigma)
    #     print(planner.target_pose)
