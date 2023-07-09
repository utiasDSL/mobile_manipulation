#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg
from liegroups import SE3, SO3
from spatialmath.base import rotz

from mmseq_plan.PlanBaseClass import Planner
from mmseq_utils.transformation import *
from mmseq_utils.parsing import parse_number

class EESimplePlanner(Planner):
    def __init__(self, planner_params):
        self.name = planner_params["name"]
        self.target_pos = np.array(planner_params["target_pos"])
        self.type = "EE"
        self.ref_type = "waypoint"
        self.ref_data_type = "Vec3"
        self.frame_id = planner_params["frame_id"]

        self.started = False
        self.finished = False
        self.reached_target = False
        self.t_reached_target = 0
        self.hold_period = planner_params["hold_period"]
        self.tracking_err_tol = planner_params["tracking_err_tol"]

        super().__init__()

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
        self.name = planner_params["name"]
        self.target_pos = np.array(planner_params["target_pos"])
        self.type = "EE"
        self.ref_type = "waypoint"
        self.ref_data_type = "Vec3"
        self.frame_id = planner_params["frame_id"]

        self.started = False
        self.finished = False
        self.reached_target = False
        self.t_reached_target = 0
        self.hold_period = planner_params["hold_period"]
        self.tracking_err_tol = planner_params["tracking_err_tol"]

        super().__init__()

    def getTrackingPoint(self, t, robot_states=None):
        q,_ = robot_states
        target_pos = self.target_pos.copy()
        target_pos[:2] -= q[:2]
        Rwb = rotz(q[2])
        target_pos= Rwb.T @ target_pos
        print(target_pos)

        return target_pos, np.zeros(3)
    
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
        self.name = config["name"]
        self.type = "EE"
        self.ref_type = "trajectory"
        self.ref_data_type = "Vec3"
        self.tracking_err_tol = config["tracking_err_tol"]
        self.frame_id = config["frame_id"]

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

        super().__init__()

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
        self.started = False

class EESimplePlannerRandom(Planner):
    def __init__(self, planner_params):
        self.target_pose_true = np.array(planner_params["target_pose"])
        self.regenerate_count = 0
        self.regenerate(0.75)
        self.type = "EE"
        self.ref_type = "pos"
        self.finished = False
        self.reached_target = False
        self.stamp = 0
        self.hold_period = planner_params["hold_period"]
        
    def getTrackingPoint(self, t, robot_states=None):
        if not self.finished:
            return self.target_pose, np.zeros(3)
        else:
            return None, None
    
    def checkFinished(self, t, current_EE):
        if self.regenerate_count == 9 and not self.reached_target and np.linalg.norm(current_EE - self.target_pose) < 0.015:
            self.reached_target = True
            self.stamp=t
            self.py_logger.info("Reached")
            
        elif self.reached_target and not self.finished:
            if t - self.stamp > self.hold_period:
                self.finished = True
                self.py_logger.info("Finished")
    def reset(self):
        self.reached_target = False
        self.stamp = 0
        self.finished = False
    
    def regenerate(self, sigma=1):
        while True:
            noise = np.random.randn(3) * sigma
            self.target_pose = self.target_pose_true +noise
            if self.target_pose[2] > 0:
                break
        self.regenerate_count +=1


class EEPosTrajectory(Planner):
    def __init__(self, planner_params):
        self.type = "EE"
        self.finished = False
        self.reached_target = False
        self.stamp = 0
        self.hold_period = planner_params["hold_period"]
        self.ref_type = "pose"

        super().__init__()
        
    def getTrackingPoint(self, t, robot_states=None):
        wp = np.array([np.sin(t), 0, np.sin(t)*np.cos(t)])
        disp = np.array([0., 2., 1.])
        
        return wp+disp, np.zeros(3)
        
    
    def checkFinished(self, t, state_ee):
        pass
    
    def reset(self):
        self.reached_target = False
        self.stamp = 0
        self.finished = False

class EESixDofWaypoint(Planner):
    def __init__(self, planner_params):
        self.target_pose = np.array(planner_params["target_pose"])
        self.type = "EE"
        self.finished = False
        self.reached_target = False
        self.stamp = 0
        self.hold_period = planner_params["hold_period"]
        self.ref_type = "pose"

        super().__init__()
        
    def getTrackingPoint(self, t, robot_states=None):
        if not self.finished:
            return self.target_pose, np.zeros(6)
        else:
            return self.target_pose, np.zeros(6)
    
    def checkFinished(self, t, state_ee):
        # state_ee a Homogeneous Transformation matrix
        Terr = np.matmul(linalg.inv(state_ee), self.target_pose)
        Terr = SE3(SO3(Terr[:3,:3]), Terr[:3, 3])
        twist = Terr.log()
        if not self.finished and np.linalg.norm(twist) > 0.2:
            self.reset()
        if not self.reached_target and np.linalg.norm(twist) < 0.1:
            self.reached_target = True
            self.stamp=t
            self.py_logger.info("Reached")
        elif self.reached_target and not self.finished:
            if t - self.stamp > self.hold_period:
                self.finished = True
                self.py_logger.info("Finished")
        
        # print("Target {}".format(self.target_pose))
        # print("Curret {}".format(state_ee))
    
    def reset(self):
        self.reached_target = False
        self.stamp = 0
        self.finished = False


if __name__ == '__main__':
    planner_params = {"target_pose": [0., 1.,], 'hold_period':1}
    
    planner = EESimplePlanner(planner_params)
    planner_params = {"target_pose": [0., 1., 1.], 'hold_period':1}
    planner = EESimplePlannerRandom(planner_params)
    
    T = make_trans_from_vec(np.array([0,0,1]) * np.pi/2, [1,0,0])
    
    planner_params = {"target_pose": T, 'hold_period': 0}
    planner = EESixDofWaypoint(planner_params)
    
    state_ee = make_trans_from_vec(np.array([0,0,1]) * np.pi/2*0.9, [1.,0,0])
    t = 0
    planner.checkFinished(t, state_ee)
    # sigma = 0.5
    # for i in range(6):
    #     sigma *= 0.5
    #     planner.regenerate(sigma)
    #     print(planner.target_pose)
