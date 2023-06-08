#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from mmseq_plan.PlanBaseClass import Planner
from mmseq_utils.parsing import parse_number

class BaseSingleWaypoint(Planner):

    def __init__(self, planner_params):
        self.name = planner_params["name"]
        self.target_pos = np.array(planner_params["target_pos"])
        self.tracking_err_tol = np.array(planner_params["tracking_err_tol"])
        self.type = "base"
        self.ref_type = "waypoint"
        self.ref_data_type = "Vec2"
        self.frame_id = planner_params["frame_id"]

        self.finished = False

        super().__init__()


    def getTrackingPoint(self, t, robot_states=None):
        return self.target_pos, np.zeros(2)

    def checkFinished(self, t, base_curr_pos):
        err = np.linalg.norm(self.target_pos - base_curr_pos)
        if err < self.tracking_err_tol and not self.finished:
            self.py_logger.info(self.name + " planner finished.")
            self.finished = True

        return self.finished

    def reset(self):
        self.finished = False
        self.py_logger.info(self.name + " planner reset.")

    @staticmethod
    def getDefaultParams():
        config = {}
        config["name"] = "Base Position"
        config["planner_type"] = "BaseSingleWaypoint"
        config["frame_id"] = "base"
        config["target_pos"] = [0, 0]
        config["tracking_err_tol"] = 0.02

        return config

class BasePosTrajectoryCircle(Planner):
    def __init__(self, config):
        self.name = config["name"]
        self.type = "base"
        self.ref_type = "trajectory"
        self.ref_data_type = "Vec2"
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

        self.dt = 0.01
        self.N = int(self.T * self.round / self.dt)
        self.plan = self._generatePlan()

        super().__init__()

    def _generatePlan(self):
        plan_pos = np.repeat([self.c], self.N, axis=0)

        ts = np.linspace(0, self.T * self.round, self.N)
        ptx = self.r * np.cos(self.omega * ts + self.phi)
        pty = self.r * np.sin(self.omega * ts + self.phi)

        plan_pos[:, 0] += ptx
        plan_pos[:, 1] += pty

        plan_vel = np.zeros_like(plan_pos)
        vx = -self.r * np.sin(self.omega * ts + self.phi) * self.omega
        vy = self.r * np.cos(self.omega * ts + self.phi) * self.omega
        plan_vel[:, 0] = vx
        plan_vel[:, 1] = vy

        return {"p": plan_pos, "v": plan_vel}

    def _interpolate(self, t, plan, dt):
        plan_len = len(plan["p"])
        indx = int(t / dt)
        if indx > plan_len - 2:
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

class BasePosTrajectoryLine(Planner):
    def __init__(self, config):
        self.name = config["name"]
        self.type = "base"
        self.ref_type = "trajectory"
        self.ref_data_type = "Vec2"
        self.tracking_err_tol = config["tracking_err_tol"]
        self.frame_id = config["frame_id"]

        self.finished = False
        self.started = False
        self.start_time = 0

        self.initial_pos = np.array(config["initial_pos"])
        self.final_pos = np.array(config["target_pos"])
        self.cruise_speed = config["cruise_speed"]

        self.dt = 0.01
        self.T = np.linalg.norm(self.initial_pos - self.final_pos) / self.cruise_speed
        self.plan = self._generatePlan()

        super().__init__()

    def _generatePlan(self):
        ts = np.linspace(0, self.T, int(self.T/self.dt)).reshape((-1, 1))
        n = (self.final_pos - self.initial_pos) / np.linalg.norm(self.initial_pos - self.final_pos)
        plan = n * ts * self.cruise_speed + self.initial_pos

        return plan

    def _interpolate(self, t, plan, dt):
        indx = int(t / dt)
        if indx > len(plan)-2:
            return plan[-1]
        elif indx < 0:
            return plan[0]

        p0 = plan[indx]
        p1 = plan[indx + 1]

        p = (p1 - p0) / dt * (t - indx * dt) + p0

        return p

    def getTrackingPoint(self, t, robot_states=None):

        if self.started and self.start_time == 0:
            self.start_time = t

        te = t - self.start_time

        p = self._interpolate(te, self.plan, self.dt)

        return p, np.zeros(3)

    def checkFinished(self, t, base_curr_pos):
        if np.linalg.norm(base_curr_pos - self.plan[-1]) < self.tracking_err_tol:
            self.finished = True
            self.py_logger.info(self.name + " Planner Finished")
        return self.finished

    def reset(self):
        self.finished = False
        self.started = False
        self.start_time = 0
            
            
            
        