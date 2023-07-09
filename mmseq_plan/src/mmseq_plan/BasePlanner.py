#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from mmseq_plan.PlanBaseClass import Planner, TrajectoryPlanner
from mmseq_utils.parsing import parse_number
from mmseq_utils.trajectory_generation import sqaure_wave

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

    def checkFinished(self, t, states):
        base_curr_pos = states[0][:2]
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

class BasePosTrajectoryCircle(TrajectoryPlanner):
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

        return {"t": ts, "p": plan_pos, "v": plan_vel}

    def getTrackingPoint(self, t, robot_states=None):

        if self.started and self.start_time == 0:
            self.start_time = t

        te = t - self.start_time

        p, v = self._interpolate(te, self.plan)

        return p, v

    def checkFinished(self, t, states):
        base_curr_pos = states[0][:2]

        if t - self.start_time > self.T * (self.round - 1):
            if np.linalg.norm(base_curr_pos - self.plan[0]) < self.tracking_err_tol:
                self.finished = True
                self.py_logger.info(self.name + " Planner Finished")
        return self.finished

    def reset(self):
        self.finished = False
        self.started = False

class BasePosTrajectoryLine(TrajectoryPlanner):
    def __init__(self, config):
        self.name = config["name"]
        self.type = "base"
        self.ref_type = "trajectory"
        self.ref_data_type = "Vec2"
        self.tracking_err_tol = config["tracking_err_tol"]
        self.frame_id = config["frame_id"]
        self.end_stop = config.get("end_stop", False)

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
        plan_pos = n * ts * self.cruise_speed + self.initial_pos

        plan_vel = np.tile(n * self.cruise_speed, (int(self.T/self.dt), 1))

        return {'t': ts, 'p': plan_pos, 'v': plan_vel}

    def getTrackingPoint(self, t, robot_states=None):

        if self.started and self.start_time == 0:
            self.start_time = t

        te = t - self.start_time

        p, v = self._interpolate(te, self.plan)

        return p, v

    def checkFinished(self, t, states):
        base_curr_pos = states[0][:2]
        base_curr_vel = states[1][:2]
        pos_cond = np.linalg.norm(base_curr_pos - self.plan['p'][-1]) < self.tracking_err_tol
        vel_cond = np.linalg.norm(base_curr_vel) < 1e-2
        if (not self.end_stop and pos_cond) or (self.end_stop and pos_cond and vel_cond):
            self.finished = True
            self.py_logger.info(self.name + " Planner Finished")
        return self.finished

    def reset(self):
        self.finished = False
        self.started = False
        self.start_time = 0

class BasePosTrajectorySqaureWave(TrajectoryPlanner):
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

        self.dt = 0.01
        self.plan = sqaure_wave(config["peak_pos"], config["valley_pos"], config["period"], config["round"], self.dt)


        super().__init__()

    def getTrackingPoint(self, t, robot_states=None):

        if self.started and self.start_time == 0:
            self.start_time = t

        te = t - self.start_time

        p, v = self._interpolate(te, self.plan)

        return p, v

    def checkFinished(self, t, states):
        base_curr_pos = states[0][:2]

        if np.linalg.norm(base_curr_pos - self.plan['p'][-1]) < self.tracking_err_tol:
            self.finished = True
            self.py_logger.info(self.name + " Planner Finished")
        return self.finished

    def reset(self):
        self.finished = False
        self.started = False
        self.start_time = 0

    @staticmethod
    def getDefaultParams():
        config = {}
        config["name"] = "Base Position"
        config["planner_type"] = "BasePosTrajectorySqaureWave"
        config["frame_id"] = "base"
        config["peak_pos"] = [0, 0]
        config["valley_pos"] = [0, 0]
        config["period"] = 10
        config["round"] = 1
        config["tracking_err_tol"] = 0.02

        return config