#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import logging
from mmseq_utils.trajectory_generation import interpolate
class Planner(ABC):
    def __init__(self):
        self.py_logger = logging.getLogger("Planner")
        self.name = "Planner"
        self.type = "base"              # base or EE
        self.ref_type = "waypoint"      # waypoint vs trajectory
        self.ref_data_type = "Vec2"     # Vec2 vs Vec3
        self.frame_id = "base"          # base or EE

    @abstractmethod
    def getTrackingPoint(self, t, robot_states=None):
        pass

    @staticmethod
    def getDefaultParams():
        return {}

class TrajectoryPlanner(Planner):

    def _interpolate(self, t, plan):
        p,v = interpolate(t, plan)

        return p, v