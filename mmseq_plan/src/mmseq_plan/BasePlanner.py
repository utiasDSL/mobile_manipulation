#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from mmseq_plan.PlanBaseClass import Planner, TrajectoryPlanner
from mmseq_utils.parsing import parse_number
from mmseq_utils.math import wrap_pi_scalar
from mmseq_utils.trajectory_generation import sqaure_wave

class BaseSingleWaypoint(Planner):

    def __init__(self, planner_params):
        super().__init__(name=planner_params["name"],
                         type="base",
                         ref_type="waypoint",
                         ref_data_type="Vec2",
                         frame_id=planner_params["frame_id"])

        self.target_pos = np.array(planner_params["target_pos"])
        self.tracking_err_tol = np.array(planner_params["tracking_err_tol"])

        self.finished = False



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
        super().__init__(name=config["name"],
                        type="base",
                        ref_type="trajectory",
                        ref_data_type="Vec2",
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

        self.dt = 0.01
        self.N = int(self.T * self.round / self.dt)
        self.plan = self._generatePlan()

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
        self.py_logger.info(self.name + " planner reset.")

class BasePosTrajectoryLine(TrajectoryPlanner):
    def __init__(self, config):
        super().__init__(name=config["name"],
                        type="base",
                        ref_type="trajectory",
                        ref_data_type="Vec2",
                        frame_id=config["frame_id"])

        self.tracking_err_tol = config["tracking_err_tol"]
        self.end_stop = config.get("end_stop", False)

        self.finished = False
        self.started = False
        self.start_time = 0

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

    def checkFinished(self, t, states):
        base_curr_pos = states[0][:2]
        base_curr_vel = states[1][:2]
        pos_cond = np.linalg.norm(base_curr_pos - self.plan['p'][-1]) < self.tracking_err_tol
        vel_cond = np.linalg.norm(base_curr_vel) < 1e-2
        if (not self.end_stop and pos_cond) or (self.end_stop and pos_cond and vel_cond):
            self.finished = True
            self.py_logger.info(self.name + " Planner Finished Position error {}".format(np.linalg.norm(base_curr_pos - self.plan['p'][-1])))
        return self.finished

    def reset(self):
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

class BasePoseTrajectoryLine(TrajectoryPlanner):
    def __init__(self, config):
        super().__init__(name=config["name"],
                        type="base",
                        ref_type="trajectory",
                        ref_data_type="Vec3",
                        frame_id=config["frame_id"])
        self.tracking_err_tol = config["tracking_err_tol"]
        self.end_stop = config.get("end_stop", False)

        self.finished = False
        self.started = False
        self.start_time = 0

        self.initial_pose = np.array(config["initial_pose"])
        self.target_pose = np.array(config["target_pose"])
        self.cruise_speed = config["cruise_speed"]
        self.yaw_speed = config["yaw_speed"]

        self.dt = 0.01
        self.plan = self._generatePlan()

    def regeneratePlan(self):
        self.plan = self._generatePlan()
        self.start_time = 0

    def _generatePlan(self):
        initial_pos = self.initial_pose[:2]
        target_pos = self.target_pose[:2]
        initial_heading = wrap_pi_scalar(self.initial_pose[2])
        target_heading = wrap_pi_scalar(self.target_pose[2])

        T_pos = np.linalg.norm(initial_pos - target_pos) / self.cruise_speed

        ts = np.linspace(0, T_pos, int(T_pos/self.dt)).reshape((-1, 1))
        n = (target_pos - initial_pos) / np.linalg.norm(initial_pos - target_pos)
        plan_pos = n * ts * self.cruise_speed + initial_pos

        plan_vel = np.tile(n * self.cruise_speed, (int(T_pos/self.dt), 1))

        heading_diff = target_heading - initial_heading
        print("target_heading: {}, initial_heading: {}".format(target_heading, initial_heading))
        # if heading_diff > np.pi:
        #     heading_diff -= 2 * np.pi
        # elif heading_diff < -np.pi:
        #     heading_diff += 2 * np.pi

        omega = np.sign(heading_diff) * self.yaw_speed
        if omega != 0:
            T_heading = heading_diff / omega
        else:
            T_heading = T_pos
        ts_heading = np.linspace(0, T_heading, int(T_heading/self.dt)).reshape((-1, 1))
        plan_heading = omega * ts_heading + initial_heading
        plan_omega = np.ones_like(plan_heading) * omega

        d = plan_pos.shape[0] - plan_heading.shape[0]
        if d > 0:
            padding = np.ones((d, 1)) * plan_heading[-1]
            plan_heading = np.vstack((plan_heading, padding))
            plan_omega = np.vstack((plan_omega, np.zeros((d, 1))))

            t = ts

        else:
            d = -d
            padding = np.ones((d, 2)) * plan_pos[-1]
            plan_pos = np.vstack((plan_pos, padding))
            plan_vel = np.vstack((plan_vel, np.zeros((d, 2))))

            t = ts_heading

        p = np.hstack((plan_pos, plan_heading))
        v = np.hstack((plan_vel, plan_omega))


        return {'t': t, 'p': p, 'v': v}

    def getTrackingPoint(self, t, robot_states=None):

        if self.started and self.start_time == 0:
            self.start_time = t

        te = t - self.start_time

        p, v = self._interpolate(te, self.plan)
        return p, v

    def checkFinished(self, t, states):
        base_curr_pose = states[0]
        base_curr_vel = states[1]
        pos_cond = np.linalg.norm(base_curr_pose - self.plan['p'][-1]) < self.tracking_err_tol
        vel_cond = np.linalg.norm(base_curr_vel) < 1e-2
        if (not self.end_stop and pos_cond) or (self.end_stop and pos_cond and vel_cond):
            self.finished = True
            self.py_logger.info(self.name + " Planner Finished Pose error {}".format(np.linalg.norm(base_curr_pose - self.plan['p'][-1])))
        return self.finished

    def reset(self):
        self.finished = False
        self.start_time = 0
        self.py_logger.info(self.name + " planner reset.")

    @staticmethod
    def getDefaultParams():
        config = {}
        config["name"] = "Base Pose"
        config["planner_type"] = "BasePoseTrajectoryLine"
        config["frame_id"] = "base"
        config["initial_pose"] = [0, 0, 0]
        config["target_pose"] = [0, 0, 0]
        config["cruise_speed"] = 0.5
        config["yaw_speed"] = 0.5
        config["tracking_err_tol"] = 0.02

        return config

class BasePosTrajectorySqaureWave(TrajectoryPlanner):
    def __init__(self, config):
        super().__init__(name=config["name"],
                        type="base",
                        ref_type="trajectory",
                        ref_data_type="Vec2",
                        frame_id=config["frame_id"])
        self.tracking_err_tol = config["tracking_err_tol"]

        self.finished = False
        self.started = False
        self.start_time = 0

        self.dt = 0.01
        self.plan = sqaure_wave(config["peak_pos"], config["valley_pos"], config["period"], config["round"], self.dt)

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