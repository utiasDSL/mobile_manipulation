#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg
# from liegroups import SE3, SO3
from spatialmath.base import rotz
from spatialmath import SE3

from mmseq_plan.PlanBaseClass import Planner
from mmseq_plan.BasePlanner import TrajectoryPlanner

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
    
    def checkFinished(self, t, ee_curr_pos):
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
    
    def checkFinished(self, t, ee_curr_pos):
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

    def checkFinished(self, t, ee_curr_pos):
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

    def checkFinished(self, t, ee_curr_pos):
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

# class EESixDofWaypoint(Planner):
#     def __init__(self, planner_params):
#         self.target_pose = np.array(planner_params["target_pose"])
#         self.type = "EE"
#         self.finished = False
#         self.reached_target = False
#         self.stamp = 0
#         self.hold_period = planner_params["hold_period"]
#         self.ref_type = "pose"

#         super().__init__()
        
#     def getTrackingPoint(self, t, robot_states=None):
#         if not self.finished:
#             return self.target_pose, np.zeros(6)
#         else:
#             return self.target_pose, np.zeros(6)
    
#     def checkFinished(self, t, state_ee):
#         # state_ee a Homogeneous Transformation matrix
#         Terr = np.matmul(linalg.inv(state_ee), self.target_pose)
#         # Terr = SE3(SO3(Terr[:3,:3]), Terr[:3, 3])
#         Terr = SE3(Terr)
#         # twist = Terr.log()
#         twist = Terr.twist()

#         if not self.finished and np.linalg.norm(twist) > 0.2:
#             self.reset()
#         if not self.reached_target and np.linalg.norm(twist) < 0.1:
#             self.reached_target = True
#             self.stamp=t
#             self.py_logger.info("Reached")
#         elif self.reached_target and not self.finished:
#             if t - self.stamp > self.hold_period:
#                 self.finished = True
#                 self.py_logger.info("Finished")
        
#         # print("Target {}".format(self.target_pose))
#         # print("Curret {}".format(state_ee))
    
#     def reset(self):
#         self.reached_target = False
#         self.stamp = 0
#         self.finished = False


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
