import logging
from abc import ABC, abstractmethod

import numpy as np

from mm_utils.trajectory_generation import interpolate


class Planner(ABC):
    """Base class for planners"""

    def __init__(self, name, type, ref_type, ref_data_type, frame_id):
        self.py_logger = logging.getLogger("Planner")
        self.name = name
        # The following variables are for automatically
        # (1) publishing rviz visualization data
        # (2) assigning the correct mpc cost function
        self.type = type  # base or EE
        self.ref_type = ref_type  # waypoint vs trajectory vs path
        self.ref_data_type = ref_data_type  # Vec2 vs Vec3
        self.frame_id = frame_id  # base or EE
        self.robot_states = None
        self.close_to_finish = False

    @abstractmethod
    def getTrackingPoint(self, t, robot_states=None):
        """get tracking point for controllers

        :param t: time (s)
        :type t: float
        :param robot_states: (joint angle, joint velocity), defaults to None
        :type robot_states: tuple, optional

        :return: position, velocity
        :rtype: numpy array, numpy array
        """
        p = None
        v = None
        return p, v

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

    def updateRobotStates(self, robot_states):
        """update robot states

        :param robot_states: (joint angle, joint velocity)
        :type robot_states: tuple
        """
        self.robot_states = robot_states

    def ready(self):
        """
        :return: true if the planner is ready to be called getTrackingPoint
        :rtype: boolean
        """
        return True

    def closeToFinish(self):
        """
        :return: true if the planner is close to finish
        :rtype: boolean
        """
        return self.close_to_finish

    def activate(self):
        self.started = True
        return


class TrajectoryPlanner(Planner):
    """Base class for trajectory planners that interpolate along a trajectory"""

    def __init__(self, name, type, ref_type, ref_data_type, frame_id):
        super().__init__(name, type, ref_type, ref_data_type, frame_id)

    def _interpolate(self, t, plan):
        p, v = interpolate(t, plan)
        return p, v

    def ready(self):
        """
        :return: true if the planner is ready to be called getTrackingPoint
        :rtype: boolean
        """
        return True

    def getTrackingPointArray(self):
        return np.zeros(0), np.zeros(0)
