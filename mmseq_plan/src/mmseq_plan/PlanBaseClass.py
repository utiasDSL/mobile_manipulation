#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import logging
from mmseq_utils.trajectory_generation import interpolate
class Planner(ABC):
    def __init__(self, name, type, ref_type, ref_data_type, frame_id):
        self.py_logger = logging.getLogger("Planner")
        self.name = name
        self.type = type                        # base or EE
        self.ref_type = ref_type                # waypoint vs trajectory
        self.ref_data_type = ref_data_type      # Vec2 vs Vec3
        self.frame_id = frame_id                # base or EE


    @abstractmethod
    def getTrackingPoint(self, t, robot_states=None):
        pass


class TrajectoryPlanner(Planner):

    def _interpolate(self, t, plan):
        p,v = interpolate(t, plan)

        return p, v