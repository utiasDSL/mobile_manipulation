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
        plan = np.repeat([self.c], self.N, axis=0)

        ts = np.linspace(0, self.T * self.round, self.N)
        ptx = self.r * np.cos(self.omega * ts + self.phi)
        pty = self.r * np.sin(self.omega * ts + self.phi)

        plan[:, 0] += ptx
        plan[:, 1] += pty

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

    def checkFinished(self, t, ee_curr_pos):
        if t - self.start_time > self.T * (self.round - 1):
            if np.linalg.norm(ee_curr_pos - self.plan[0]) < self.tracking_err_tol:
                self.finished = True
                self.py_logger.info(self.name + " Planner Finished")
        return self.finished

    def reset(self):
        self.finished = False
        self.started = False


class BaseSimplePlanner(Planner):
    
    def __init__(self, planner_params):
        self.init_pose = np.array(planner_params["init_pose"])
        self.target_pose = np.array(planner_params["target_pose"])
        self.T = planner_params["T"]
        self.p_norm = (self.target_pose - self.init_pose)/np.linalg.norm(self.target_pose - self.init_pose)
        self.type = "base"
        self.finished = False
        self.start = False
        self.start_time = 0
        self.cost_type = "BasePos2"

        super().__init__()
        
    def getTrackingPoint(self, t, robot_states=None):
        if self.start is False:
            self.start = True
            self.start_time = t
        
        # if t - self.start_time < self.T:
        if False:
            s = 0.5*np.sin(np.pi/self.T*t - np.pi/2) + 0.5
            p = (1 - s) * self.init_pose + s * self.target_pose
            v = self.p_norm * (0.5* np.pi/self.T*np.cos(np.pi/self.T*t - np.pi/2))
        else:
            p = self.target_pose
            v = np.zeros(2)
        return p, v 
    
    
    def checkFinished(self, t, pose):
        err = np.linalg.norm(self.target_pose - pose)
        if err < 0.1 and not self.finished:
            self.py_logger.info("base_finished")
            self.finished = True
            

class BaseArcPlanner(Planner):
    
    def __init__(self, planner_params):
        self.c = np.array(planner_params["c"])
        self.P0 = np.array(planner_params["init_point"])
        self.theta = planner_params["theta"]
        self.omega = planner_params["omega"]
        self.t = 0
        
        R = self.getRz(self.theta)
        self.target_pos = self.c + np.matmul(R, self.P0 - self.c)
        
        self.start = False
        self.type = "base"
        self.finished = False

        super().__init__()
        
    def getTrackingPoint(self, t, robot_states=None):
        if not self.start:
            self.t = t
            self.start = True
            
        phi = (t - self.t) * self.omega
        if self.omega > 0:
            phi = min(self.theta, phi)
        if self.omega < 0:
            phi = max(self.theta, phi)
        
        R = self.getRz(phi)
        p = self.c + np.matmul(R, self.P0 - self.c)
        omegax = np.array([[0, -self.omega, 0], 
                           [self.omega, 0, 0.],
                           [-0., 0., 0]])
        v = np.matmul(omegax, p - self.c)
        p[2] = np.arctan2(v[1], v[0])    
        if p[2] > np.pi/2:
            p[2] -= 2*np.pi
        v[2] = self.omega*0.2
        return p, np.zeros(3)
    
    def getRz(self, phi):
        R = np.zeros((3,3))
        R[0,0] = np.cos(phi)
        R[1,0] = np.sin(phi)
        R[0,1] = -np.sin(phi)
        R[1,1] = np.cos(phi)
        R[2,2] = 1
        
        return R
    
    def checkFinished(self, t, pose):
        err = np.linalg.norm(self.target_pos[:2] - pose[:2])
        if err < 0.4:
            self.finished = True


if __name__ == '__main__':
    getPtCircle = lambda  c, r, theta: [c[0] + r*np.cos(theta), c[1] + r*np.sin(theta), 1.2]
    c = [2., 0.]
    r = 1
    theta = [5*np.pi/6, np.pi/6, -np.pi/2]
    ee_target = [getPtCircle(c, r, theta[0]), getPtCircle(c, r, theta[1]), getPtCircle(c, r, theta[2])]
    rb = r+1
    base_target = [getPtCircle(c, rb, theta[0]), getPtCircle(c, rb, theta[1]), getPtCircle(c, rb, theta[2])]
    
    base_params = {"c": c+[1.2], "init_point":base_target[1], "theta": theta[2]- theta[1], "omega":-np.pi/3}
    planner = BaseArcPlanner(base_params)
    
    traj_p = []
    traj_v = []
    time = np.arange(300) * 0.01
    for t in time:
        p,v = planner.getTrackingPoint(t)
        traj_p.append(p)
        traj_v.append(v)
        
    from matplotlib import pyplot as plt
    traj_p = np.array(traj_p)
    
    fig, axis = plt.subplots(1, figsize=[10,10])
    plt.plot(traj_p[:,0], traj_p[:, 1], '-',label="arc")

    plt.legend()
    plt.show()
            
            
            
        