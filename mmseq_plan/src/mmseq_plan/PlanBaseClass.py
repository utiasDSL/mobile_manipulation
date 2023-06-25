#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import logging
from mmseq_utils.trajectory_generation import interpolate
class Planner(ABC):
    def __init__(self):
        self.py_logger = logging.getLogger("Planner")

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