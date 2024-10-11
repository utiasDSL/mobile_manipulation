#!/usr/bin/env python3

from abc import ABC, abstractmethod

import numpy as np
import casadi as cs
import time
import logging
import copy 

from typing import Optional, List, Dict, Tuple, Union
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from mmseq_plan.PlanBaseClass import Planner,TrajectoryPlanner
from mmseq_utils.math import wrap_pi_array
from mmseq_utils.casadi_struct import casadi_sym_struct
from mmseq_control_new.MPCConstraints import HierarchicalTrackingConstraint
from mmseq_control_new.MPCCostFunctions import CostFunctions
import mobile_manipulation_central as mm

from mmseq_control_new.MPC import MPC, INF
import mmseq_control_new.MPC as MPC_Mod
import mmseq_control_new.HTMPC as HTMPC_Mod
from mmseq_utils.parsing import load_config, parse_ros_path, recursive_dict_update
import mobile_manipulation_central as mm

class HybridMPC():

    def __init__(self, config_in) -> None:
        self.controllers = {}
        self.logs = {}
        config = copy.deepcopy(config_in)
        config.pop("type")
        control_modes = config.pop("control_modes")

        for path in control_modes:
            config_mode = load_config(parse_ros_path(path))
            name = path["key"]
            config_mode["controller"] = recursive_dict_update(copy.deepcopy(config), config_mode["controller"])

            ctrl_config = config_mode["controller"]
            control_class = getattr(HTMPC_Mod, ctrl_config["type"], None)
            if control_class is None:
                control_class = getattr(MPC_Mod, ctrl_config["type"], None)

            self.controllers[name] = control_class(ctrl_config)
            self.logs[name] = self.controllers[name].log

            config[name] = ctrl_config
            print("Add controller {}".format(name))
    
        self.params = config

        model_controller = self.controllers[name]
        self.model_interface = model_controller.model_interface
        self.robot = self.model_interface.robot
        self.ssSymMdl = self.robot.ssSymMdl
        self.kinSymMdl = self.robot.kinSymMdls

        self.nx = self.ssSymMdl["nx"]
        self.nu = self.ssSymMdl["nu"]
        self.DoF = self.robot.DoF
        self.home = mm.load_home_position(config.get("home", "default"))


        self.dt = self.params["dt"]
        self.tf = self.params['prediction_horizon']
        self.N = int(self.tf / self.dt)
        self.QPsize = self.nx * (self.N + 1) + self.nu * self.N

        self.x_bar = np.zeros((self.N + 1, self.nx))  # current best guess x0,...,xN
        self.x_bar[:, :self.DoF] = self.home
        self.u_bar = np.zeros((self.N, self.nu))  # current best guess u0,...,uN-1
        self.t_bar = None
        self.zopt = np.zeros(self.QPsize)  # current linearization point
        self.u_prev = np.zeros(self.nu)
        self.xu_bar_init = False

        self.v_cmd = np.zeros(self.nx - self.DoF)

        self.py_logger = logging.getLogger("Controller")

        self.ree_bar = None
        self.rbase_bar = None
        self.ee_bar = None
        self.base_bar = None
        self.sdf_bar = {"EE":None,
                        "base": None}
        self.sdf_grad_bar = {"EE":None,
                            "base": None}
        
        self.log = self._get_log()
        
    @abstractmethod
    def control(self, 
                t: float, 
                robot_states: Tuple[NDArray[np.float64], NDArray[np.float64]], 
                planners: List[Union[Planner, TrajectoryPlanner]], 
                map=None):
        
        pass
    

    def _get_log(self):
        log = {"curr_controller": ""}
        for name in self.controllers.keys():
            log[name] = {}
        
        return log

class RAL25(HybridMPC):

    def __init__(self, config) -> None:
        super().__init__(config)
        self.prev_controller_name = None
    
    def control(self, 
                t: float, 
                robot_states: Tuple[NDArray[np.float64], NDArray[np.float64]], 
                planners: List[Union[Planner, TrajectoryPlanner]], 
                map=None):
        curr_controller_name = "HTMPC"
        for planner in planners:
            print(planners)
            if "baseframe" in planner.name.split("_"):
                curr_controller_name = "NavMPC"
        self.py_logger.info("Acting Controller {}".format(curr_controller_name))
        controller = self.controllers[curr_controller_name] 

        if self.prev_controller_name is not None and self.prev_controller_name != curr_controller_name:
            controller.reset()
            # pass in xu bar 
            # controller.u_bar = np.zerso_like(controller.u_bar)
            # controller.x_bar = self.x_bar.copy()
            # controller.t_bar = self.t_bar.copy()
            # controller.lam_bar = None


        results = controller.control(t, robot_states, planners, map)
        self._copy_internal_states(controller)
        self._log(controller, curr_controller_name)
        self.prev_controller_name = curr_controller_name

        return results

    
    def _copy_internal_states(self, controller: MPC):

        self.x_bar = controller.x_bar.copy()
        self.u_bar = controller.u_bar.copy()
        self.t_bar = controller.t_bar.copy()
        self.ree_bar = controller.ree_bar.copy()
        self.rbase_bar = controller.rbase_bar.copy()
        self.sdf_bar = copy.deepcopy(controller.sdf_bar)
        self.sdf_grad_bar = copy.deepcopy(controller.sdf_grad_bar)

        self.ee_bar, self.base_bar = controller.ee_bar.copy(), controller.base_bar.copy()
    
    def _log(self, controller, controller_name):
        self.log["curr_controller"] = controller_name
        self.log[controller_name] = copy.deepcopy(controller.log)


