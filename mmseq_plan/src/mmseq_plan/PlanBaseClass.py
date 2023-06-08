#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import logging

class Planner(ABC):
    def __init__(self):
        self.py_logger = logging.getLogger("Planner")

    @abstractmethod
    def getTrackingPoint(self, t, robot_states=None):
        pass

    @staticmethod
    def getDefaultParams():
        return {}
