#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import logging
from mmseq_utils.trajectory_generation import interpolate
class Planner(ABC):
    def __init__(self, name, type, ref_type, ref_data_type, frame_id):
        self.py_logger = logging.getLogger("Planner")
        self.name = name
        # The following variables are for automatically 
        # (1) publishing rviz visualization data
        # (2) assigning the correct mpc cost function
        self.type = type                        # base or EE
        self.ref_type = ref_type                # waypoint vs trajectory
        self.ref_data_type = ref_data_type      # Vec2 vs Vec3
        self.frame_id = frame_id                # base or EE


    @abstractmethod
    def getTrackingPoint(self, t, robot_states=None):
        """ get tracking point for controllers

        :param t: time (s)
        :type t: float
        :param robot_states: (joint angle, joint velocity), defaults to None
        :type robot_states: tuple, optional
        
        :return: position, velocity
        :rtype: numpy array, numpy array
        """
        p = None
        v = None
        return p,v
    
    @abstractmethod
    def checkFinished(self, t, P):
        """check if the planner is finished 

        :param t: time since the controller started
        :type t: float
        :param P: EE position for EE planner, base position for base planner
        :type P: numpy array
        :return: true if the planner has finished, false otherwise
        :rtype: boolean
        """
        finished = True
        return finished

class TrajectoryPlanner(Planner):

    def _interpolate(self, t, plan):
        p,v = interpolate(t, plan)

        return p, v

class WholeBodyPlanner(Planner):
    def __init__(self):
        pass
    
    @abstractmethod
    def generatePlanFromConfig(self):
        pass

    @abstractmethod
    def interpolate(self, t):
        pass

    @abstractmethod
    def processResults(self):
        pass

    @abstractmethod
    def initiliazePartialPlanners(self):
        pass
    
    def control(self, t):
        ''' Given a time, return an interpolated velocity'''
        q, q_dot = self.interpolate(t)
        return q_dot
    
    def getTrackingPoint(self, t):
        ''' Given a time, return the end effector position and base position'''
        q, q_dot = self.interpolate(t)

        end_effector_p = self.motion_class.end_effector_pose(q)
        end_effector_v = self.motion_class.compute_jacobian_whole(q) @ q_dot

        base_p = self.motion_class.base_xyz(q)
        base_v = self.motion_class.base_jacobian(q) @ q_dot
        return (end_effector_p, end_effector_v), (base_p, base_v)
    
    def checkFinished(self, t, ee_curr_pos):
        return t > self.tf

    def getBasePlanner(self):
        return self.base_planner
    
    def getEEPlanner(self):
        return self.ee_planner
        

class CasadiPartialPlanner(Planner):
    def __init__(self, config, pose_calculator_func, jacobian_calculator_func):
        self.name = config["name"]
        self.type = config["type"]
        self.ref_type = "trajectory"
        self.ref_data_type = config["ref_data_type"]
        self.tracking_err_tol = config["tracking_err_tol"]
        self.frame_id = config["frame_id"]

        self.pose_calculator_func = pose_calculator_func
        self.jacobian_calculator_func = jacobian_calculator_func

    @abstractmethod
    def interpolate(self, t):
        pass

    @abstractmethod
    def processResults(self):
        pass

    def getTrackingPoint(self, t, robot_states=None):
        ''' Given a time, return the end effector position and base position'''
        q, q_dot = self.interpolate(t)

        p = self.pose_calculator_func(q)
        v = self.jacobian_calculator_func(q) @ q_dot

        return p.full().flatten(), v.full().flatten()
    
    def checkFinished(self, t, ee_curr_pos):
        return t > self.tf

    

