#!/usr/bin/env python3

from abc import ABC, abstractmethod
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from numpy.typing import NDArray

import numpy as np
import casadi as cs
from spatialmath.base import rotz
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from scipy.interpolate import interp1d

from mmseq_control.robot import MobileManipulator3D as MM
from mmseq_control.robot import CasadiModelInterface as ModelInterface
from mmseq_plan.PlanBaseClass import Planner,TrajectoryPlanner

from mmseq_utils.math import wrap_pi_array
from mmseq_utils.casadi_struct import casadi_sym_struct
from mmseq_utils.parsing import parse_ros_path, parse_path
import mmseq_control_new.MPCConstraints as MPCConstraints
from mmseq_control_new.MPCCostFunctions import *
from mmseq_control_new.MPCConstraints import SignedDistanceConstraint, StateBoxConstraints, ControlBoxConstraints
import mobile_manipulation_central as mm
INF = 1e5

class MPC():
    def __init__(self, config):
        self.model_interface = ModelInterface(config)
        self.robot = self.model_interface.robot
        self.ssSymMdl = self.robot.ssSymMdl
        self.kinSymMdl = self.robot.kinSymMdls
        self.nx = self.ssSymMdl["nx"]
        self.nu = self.ssSymMdl["nu"]
        self.DoF = self.robot.DoF
        self.home = mm.load_home_position(config.get("home", "default"))

        self.params = config
        self.dt = self.params["dt"]
        self.tf = self.params['prediction_horizon']
        self.N = int(self.tf / self.dt)
        self.QPsize = self.nx * (self.N + 1) + self.nu * self.N

        self.EEPos3Cost = EEPos3CostFunction(self.robot, config["cost_params"]["EEPos3"])
        self.EEPos3BaseFrameCost = EEPos3BaseFrameCostFunction(self.robot, config["cost_params"]["EEPos3"])

        self.EEPoseSE3Cost = EEPoseSE3CostFunction(self.robot, config["cost_params"]["EEPose"])
        self.EEPoseBaseFrameSE3Cost = EEPoseSE3BaseFrameCostFunction(self.robot, config["cost_params"]["EEPose"])

        self.BasePos2Cost = BasePos2CostFunction(self.robot, config["cost_params"]["BasePos2"])
        self.BasePos3Cost = BasePos3CostFunction(self.robot, config["cost_params"]["BasePos3"])
        self.BasePoseSE2Cost = BasePoseSE2CostFunction(self.robot, config["cost_params"]["BasePoseSE2"])

        self.EEVel3Cost = EEVel3CostFunction(self.robot, config["cost_params"]["EEVel3"])
        self.BaseVel2Cost = BaseVel2CostFunction(self.robot, config["cost_params"]["BaseVel2"])
        self.BaseVel3Cost = BaseVel3CostFunction(self.robot, config["cost_params"]["BaseVel3"])

        self.CtrlEffCost = ControlEffortCostFunction(self.robot, config["cost_params"]["Effort"])
        self.RegularizationCost = RegularizationCostFunction(self.nx, self.nu)
        
        self.collision_link_names = ["self"] if self.params["self_collision_avoidance_enabled"] else []
        self.collision_link_names += self.model_interface.scene.collision_link_names["static_obstacles"] \
            if self.params["static_obstacles_collision_avoidance_enabled"] else []
        self.collision_link_names += ["sdf"] if self.params["sdf_collision_avoidance_enabled"] else []

        self.collisionCsts = {}
        for name in self.collision_link_names:
            sd_fcn = self.model_interface.getSignedDistanceSymMdls(name)
            if name in self.model_interface.scene.collision_link_names["static_obstacles"]:
                collision_cst_type = getattr(MPCConstraints, self.params["collision_constraint_type"]["static_obstacles"])
                sd_cst = collision_cst_type(self.robot, sd_fcn,
                                            self.params["collision_safety_margin"]["static_obstacles"], name)
            else:
                collision_cst_type = getattr(MPCConstraints, self.params["collision_constraint_type"][name])

                sd_cst = collision_cst_type(self.robot, sd_fcn,
                                            self.params["collision_safety_margin"][name], name)
            self.collisionCsts[name] = sd_cst
            
        self.collisionSoftCsts = {}
        for name,sd_cst in self.collisionCsts.items():
            expand = True if name !="sdf" else False
            if name in self.model_interface.scene.collision_link_names["static_obstacles"]:
                self.collisionSoftCsts[name] = SoftConstraintsRBFCostFunction(self.params["collision_soft"]["static_obstacles"]["mu"],
                                                                            self.params["collision_soft"]["static_obstacles"]["zeta"],
                                                                            sd_cst, name+"CollisionSoftCst",
                                                                            expand=expand)
            else:
                self.collisionSoftCsts[name] = SoftConstraintsRBFCostFunction(self.params["collision_soft"][name]["mu"],
                                                                            self.params["collision_soft"][name]["zeta"],
                                                                            sd_cst, name+"CollisionSoftCst",
                                                                            expand=expand)
        
        self.stateCst = StateBoxConstraints(self.robot)
        self.controlCst = ControlBoxConstraints(self.robot)

        self.cost = []
        self.constraints = []

        self.x_bar = np.zeros((self.N + 1, self.nx))  # current best guess x0,...,xN
        self.x_bar[:, :self.DoF] = self.home
        self.u_bar = np.zeros((self.N, self.nu))  # current best guess u0,...,uN-1
        self.t_bar = None
        self.lam_bar = None                       # inequality mulitipliers
        self.zopt = np.zeros(self.QPsize)  # current lineaization point
        self.u_prev = np.zeros(self.nu)
        self.xu_bar_init = False

        self.v_cmd = np.zeros(self.nx - self.DoF)

        self.py_logger = logging.getLogger("Controller")
        self.log = self._get_log()

        self.ree_bar = None
        self.rbase_bar = None
        self.ee_bar = None
        self.base_bar = None
        self.sdf_bar = {"EE":None,
                        "base": None}
        self.sdf_grad_bar = {"EE":None,
                            "base": None}

        self.output_dir = Path(parse_ros_path({"package": "mmseq_control_new", "path":"acados_outputs"}))
        if not self.output_dir.exists():
            self.output_dir.mkdir()

    @abstractmethod
    def control(self, 
                t: float, 
                robot_states: Tuple[NDArray[np.float64], NDArray[np.float64]], 
                planners: List[Union[Planner, TrajectoryPlanner]], 
                map=None):
        """

        :param t: current control time
        :param robot_states: (q, v) generalized coordinates and velocities
        :param planners: a list of planner instances
        :return: u, currently the best control inputs, aka, u_bar[0]
        """
        pass

    def _predictTrajectories(self, xo, u_bar):
        return MM.ssIntegrate(self.dt, xo, u_bar, self.ssSymMdl)

    def _getEEBaseTrajectories(self, x_bar):
        ee_bar = np.zeros((self.N + 1, 3))
        base_bar = np.zeros((self.N + 1, 3))
        for k in range(self.N+1):
            base_bar[k] = x_bar[k, :3]
            fee_fcn = self.kinSymMdl[self.robot.tool_link_name]
            ee_pos, ee_orn = fee_fcn(x_bar[k, :self.DoF])
            ee_bar[k] = ee_pos.toarray().flatten()

        return ee_bar, base_bar


    def reset(self):
        self.x_bar = np.zeros((self.N + 1, self.nx))  # current best guess x0,...,xN
        self.u_bar = np.zeros((self.N, self.nu))  # current best guess u0,...,uN-1
        self.t_bar = None
        self.lam_bar = None
        self.zopt = np.zeros(self.QPsize)  # current lineaization point
        self.xu_bar_init = False

        self.v_cmd = np.zeros(self.nx - self.DoF)
    
    def evaluate_cost_function(self, cost_function: CostFunctions, x_bar, u_bar, nlp_p_map_bar):
        ps = []
        cost_p_dict = cost_function.get_p_dict()
        cost_p_struct = casadi_sym_struct(cost_p_dict)
        cost_p_map = cost_p_struct(0)

        vals = []
        for k in range(self.N):
            nlp_p_map = nlp_p_map_bar[k]
            for key in cost_p_map.keys():
                cost_p_map[key] = nlp_p_map[key]
            v = cost_function.evaluate(x_bar[k], u_bar[k], cost_p_map.cat.full().flatten())
            vals.append(v)
        return np.sum(vals)
    
    def evaluate_constraints(self, constraints: MPCConstraints.Constraint, x_bar, u_bar, nlp_p_map_bar):
        ps = []
        cost_p_dict = constraints.get_p_dict()
        cost_p_struct = casadi_sym_struct(cost_p_dict)
        cost_p_map = cost_p_struct(0)

        vals = []
        for k in range(self.N+1):
            nlp_p_map = nlp_p_map_bar[k]
            for key in cost_p_map.keys():
                cost_p_map[key] = nlp_p_map[key]
            if k < self.N:
                v = constraints.check(x_bar[k], u_bar[k], cost_p_map.cat.full().flatten())
            else:
                v = constraints.check(x_bar[k], u_bar[k-1], cost_p_map.cat.full().flatten())

            vals.append(v)
        return vals
    
    def evaluate_sdf_h_fcn(self, constraints: MPCConstraints.Constraint, x_bar, u_bar, nlp_p_map_bar):
        ps = []
        cost_p_dict = constraints.get_p_dict()
        cost_p_struct = casadi_sym_struct(cost_p_dict)
        cost_p_map = cost_p_struct(0)

        vals = []
        for k in range(self.N+1):
            nlp_p_map = nlp_p_map_bar[k]
            for key in cost_p_map.keys():
                cost_p_map[key] = nlp_p_map[key]
            if k < self.N:
                v = constraints.h_fcn(x_bar[k], u_bar[k], cost_p_map.cat.full().flatten())
            else:
                v = constraints.h_fcn(x_bar[k], u_bar[k-1], cost_p_map.cat.full().flatten())

            vals.append(v)
        return vals
    
    def evaluate_sdf_xdot_fcn(self, constraints: MPCConstraints.Constraint, x_bar, u_bar, nlp_p_map_bar):
        ps = []
        cost_p_dict = constraints.get_p_dict()
        cost_p_struct = casadi_sym_struct(cost_p_dict)
        cost_p_map = cost_p_struct(0)

        vals = []
        for k in range(self.N+1):
            nlp_p_map = nlp_p_map_bar[k]
            for key in cost_p_map.keys():
                cost_p_map[key] = nlp_p_map[key]
            if k < self.N:
                v = constraints.xdot_fcn(x_bar[k], u_bar[k], cost_p_map.cat.full().flatten())
            else:
                v = constraints.xdot_fcn(x_bar[k], u_bar[k-1], cost_p_map.cat.full().flatten())

            vals.append(v)
        return vals

    def evaluate_constraints_gradient(self, constraints: MPCConstraints.Constraint, x_bar, u_bar, nlp_p_map_bar):
        ps = []
        cost_p_dict = constraints.get_p_dict()
        cost_p_struct = casadi_sym_struct(cost_p_dict)
        cost_p_map = cost_p_struct(0)

        vals = []
        for k in range(self.N+1):
            nlp_p_map = nlp_p_map_bar[k]
            for key in cost_p_map.keys():
                cost_p_map[key] = nlp_p_map[key]
            if k < self.N:
                v = constraints.g_grad_fcn(x_bar[k], u_bar[k], cost_p_map.cat.full().flatten()).T
            else:
                v = constraints.g_grad_fcn(x_bar[k], u_bar[k-1], cost_p_map.cat.full().flatten()).T
            vals.append(v)
        return vals

    def _construct(self, costs, constraints, num_terminal_cost, name="MM"):
        # Construct AcadosModel
        model = AcadosModel()
        model.x = cs.MX.sym('x', self.nx)
        model.u = cs.MX.sym('u', self.nu)
        model.xdot = cs.MX.sym('xdot', self.nx)

        model.f_impl_expr = model.xdot - self.ssSymMdl["fmdl"](model.x, model.u)
        model.f_expl_expr = self.ssSymMdl["fmdl"](model.x, model.u)
        model.name = name

        # get params from constraints
        p_dict = {}
        for cost in costs:
            p_dict.update(cost.get_p_dict())
        for cst in constraints:
            p_dict.update(cst.get_p_dict())
        p_struct = casadi_sym_struct(p_dict)
        print(p_struct)
        p_map = p_struct(0)
        model.p = p_struct.cat

        # Construct AcadosOCP
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.tf
        ocp.code_export_directory = str(self.output_dir / "c_generated_code")
        ocp.solver_options.ext_fun_compile_flags = '-O2'

        ocp.cost.cost_type = 'EXTERNAL'
        cost_expr = []
        for cost in costs:
            Ji = cost.J_fcn(model.x, model.u, cost.p_sym)
            cost_expr.append(Ji)
        ocp.model.cost_expr_ext_cost = sum(cost_expr)

        if self.params["acados"]["use_custom_hess"]:
            custom_hess_expr = []

            for cost in costs:
                H_fcn = cost.get_custom_H_fcn()
                H_expr_i = H_fcn(model.x, model.u, cost.p_sym)
                custom_hess_expr.append(H_expr_i)
            ocp.model.cost_expr_ext_cost_custom_hess = sum(custom_hess_expr)

        if self.params["acados"]["use_terminal_cost"]:
            ocp.cost.cost_type_e = 'EXTERNAL'
            cost_expr_e = sum(cost_expr[:num_terminal_cost])
            cost_expr_e = cs.substitute(cost_expr_e, model.u, [])
            model.cost_expr_ext_cost_e = cost_expr_e
            if self.params["acados"]["use_custom_hess"]:
                cost_hess_expr_e = sum(custom_hess_expr[:num_terminal_cost])
                cost_hess_expr_e = cs.substitute(cost_hess_expr_e, model.u, [])
                model.cost_expr_ext_cost_custom_hess_e = cost_hess_expr_e

        # control input constraints
        ocp.constraints.lbu = np.array(self.ssSymMdl["lb_u"])
        ocp.constraints.ubu = np.array(self.ssSymMdl["ub_u"])
        ocp.constraints.idxbu = np.arange(self.nu)

        if self.params["acados"]["slack_enabled"]["u"]:
            ocp.constraints.idxsbu = np.arange(self.nu)
            ocp.constraints.lsbu = np.zeros(self.nu)
            ocp.constraints.usbu = np.zeros(self.nu)
            nsu = self.nu
        else:
            nsu = 0

        # state constraints
        ocp.constraints.lbx = np.array(self.ssSymMdl["lb_x"])
        ocp.constraints.ubx = np.array(self.ssSymMdl["ub_x"])
        ocp.constraints.idxbx = np.arange(self.nx)

        if self.params["acados"]["slack_enabled"]["x"]:
            ocp.constraints.idxsbx = np.arange(self.nx)
            ocp.constraints.lsbx = np.zeros(self.nx)
            ocp.constraints.usbx = np.zeros(self.nx)
            nsx = self.nx
        else:
            nsx = 0

        ocp.constraints.lbx_e = np.array(self.ssSymMdl["lb_x"])
        ocp.constraints.ubx_e = np.array(self.ssSymMdl["ub_x"])
        ocp.constraints.idxbx_e = np.arange(self.nx)

        if self.params["acados"]["slack_enabled"]["x_e"]:
            ocp.constraints.idxsbx_e = np.arange(self.nx)
            ocp.constraints.lsbx_e = np.zeros(self.nx)
            ocp.constraints.usbx_e = np.zeros(self.nx)
            nsx_e = self.nx
        else:
            nsx_e = 0

        # nonlinear constraints
        # TODO: what about the initial and terminal shooting nodes.
        h_expr_list = []
        idxsh = []
        h_idx = 0
        for i, cst in enumerate(constraints):
            h_expr_list.append(cst.g_fcn(model.x, model.u, cst.p_sym))
            if cst.slack_enabled and (self.params["acados"]["slack_enabled"]["h"] or self.params["acados"]["slack_enabled"]["h_0"] or self.params["acados"]["slack_enabled"]["h_e"]):
                idxsh += [h_i for h_i in range(h_idx, h_idx + cst.ng)]
            h_idx += cst.ng

        nsh = len(idxsh) if self.params["acados"]["slack_enabled"]["h"] else 0
        nsh_e = len(idxsh) if self.params["acados"]["slack_enabled"]["h_e"] else 0
        nsh_0 = len(idxsh) if self.params["acados"]["slack_enabled"]["h_0"] else 0
        if len(h_expr_list) > 0:
            h_expr = cs.vertcat(*h_expr_list)
            h_expr_num = h_expr.shape[0]
            print(h_expr)

            model.con_h_expr_0 = h_expr
            ocp.constraints.uh_0 = np.zeros(h_expr_num)
            ocp.constraints.lh_0 = -INF*np.ones(h_expr_num)

            model.con_h_expr = h_expr
            ocp.constraints.uh = np.zeros(h_expr_num)
            ocp.constraints.lh = -INF*np.ones(h_expr_num)

            model.con_h_expr_e = cs.substitute(h_expr, model.u, [])
            ocp.constraints.uh_e = np.zeros(h_expr_num)
            ocp.constraints.lh_e = -INF*np.ones(h_expr_num)
        else:
            h_expr_num = 0
        
        if h_expr_num > 0: 

            if nsh_0 > 0:
                ocp.constraints.idxsh_0 = np.array(idxsh)
                ocp.constraints.lsh_0 = np.zeros(nsh_0)
                ocp.constraints.ush_0 = np.zeros(nsh_0)
            if nsh > 0:
                ocp.constraints.idxsh = np.array(idxsh)
                ocp.constraints.lsh = np.zeros(nsh)
                ocp.constraints.ush = np.zeros(nsh)
            if nsh_e > 0:
                ocp.constraints.idxsh_e = np.array(idxsh)
                ocp.constraints.lsh_e  = np.zeros(nsh_e)
                ocp.constraints.ush_e  = np.zeros(nsh_e)

            
        # TODO: slack variables?
        ns = nsx + nsu + nsh
        z = self.params["cost_params"]["slack"]["z"]
        Z = self.params["cost_params"]["slack"]["Z"]

        if ns > 0:
            ocp.cost.Zl = np.ones(ns) * Z
            ocp.cost.Zu = np.ones(ns) * Z
            ocp.cost.zl = np.ones(ns) * z
            ocp.cost.zu = np.ones(ns) * z

        ns_e = nsx_e + nsh_e
        if ns_e > 0:
            ocp.cost.Zl_e = np.ones(ns_e) * Z
            ocp.cost.Zu_e = np.ones(ns_e) * Z
            ocp.cost.zl_e = np.ones(ns_e) * z
            ocp.cost.zu_e = np.ones(ns_e) * z
            
        ns_0 = nsh_0 + nsu
        if ns_0 > 0:
            ocp.cost.Zl_0 = np.ones(ns_0) * Z
            ocp.cost.Zu_0 = np.ones(ns_0) * Z
            ocp.cost.zl_0 = np.ones(ns_0) * z
            ocp.cost.zu_0 = np.ones(ns_0) * z
        # initial condition
        ocp.constraints.x0 = self.x_bar[0]

        ocp.parameter_values = p_map.cat.full().flatten()

        # set options from config
        for key, val in self.params["acados"]["ocp_solver_options"].items():
            property = getattr(ocp.solver_options, key, None)
            if property is not None:
                setattr(ocp.solver_options, key, val)
            else:
                self.py_logger.warning(f"{key} not found in Acados solver options. Parameter is ignored.")
                
        # Construct AcadosOCPSolver
        json_file_name = str(self.output_dir/f"acados_ocp_{name}.json")
        if self.params["acados"]["cython"]["enabled"]:
            if self.params["acados"]["cython"]["action"] == "generate":
                AcadosOcpSolver.generate(ocp, json_file=json_file_name)
                AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
                ocp_solver = AcadosOcpSolver.create_cython_solver(json_file_name)
            elif self.params["acados"]["cython"]["action"] == "load":
                # ctypes
                ## Note: skip generate and build assuming this is done before (in cython run)
                ocp_solver = AcadosOcpSolver(ocp, json_file=json_file_name, build=False, generate=False)
        else: 
            ocp_solver = AcadosOcpSolver(ocp, 
                                        json_file = json_file_name,
                                        build=True)

        return ocp, ocp_solver, p_struct

    def _get_log(self):
        return {}

class STMPC(MPC):

    def __init__(self, config):
        super().__init__(config)
        num_terminal_cost = 2
        if config["base_pose_tracking_enabled"]:
            costs = [self.BasePoseSE2Cost, self.BaseVel3Cost, self.EEPoseSE3Cost, self.EEVel3Cost, self.EEPoseBaseFrameSE3Cost, self.CtrlEffCost]
        else:
            costs = [self.BasePos2Cost, self.BaseVel2Cost, self.EEPos3Cost, self.EEVel3Cost, self.CtrlEffCost]
        
        constraints = []
        for name in self.collision_link_names:
            if name in self.model_interface.scene.collision_link_names["static_obstacles"]:
                softened = self.params["collision_constraints_softend"]["static_obstacles"]
            else:
                softened = self.params["collision_constraints_softend"][name]

            if softened:
                costs.append(self.collisionSoftCsts[name])
            else:
                constraints.append(self.collisionCsts[name])
        # constraints = [cst for cst in self.collisionCsts.values()]
        name = self.params["acados"].get("name", "MM")
        self.ocp, self.ocp_solver, self.p_struct = self._construct(costs, constraints, num_terminal_cost, name)

        self.cost = costs
        self.constraints = constraints + [self.controlCst, self.stateCst]

    def control(self, 
                t: float, 
                robot_states: Tuple[NDArray[np.float64], NDArray[np.float64]], 
                planners: List[Union[Planner, TrajectoryPlanner]], 
                map=None):

        self.py_logger.debug("control time {}".format(t))
        self.curr_control_time = t
        q, v = robot_states
        q[2:9] = wrap_pi_array(q[2:9])
        xo = np.hstack((q, v))

        # 0.1 Get warm start point
        # self.u_bar[:-1] = self.u_bar[1:].copy()
        # self.u_bar[-1] = 0
        # self.x_bar = self._predictTrajectories(xo, self.u_bar)
        if self.t_bar is not None:
            self.u_t = interp1d(self.t_bar, self.u_bar, axis=0, 
                                bounds_error=False, fill_value="extrapolate")
            t_bar_new = t + np.arange(self.N)* self.dt
            self.u_bar = self.u_t(t_bar_new)
            self.x_bar = self._predictTrajectories(xo, self.u_bar)
        else:
            self.u_bar = np.zeros_like(self.u_bar)
            # self.x_bar[:, 9:] = np.zeros((self.N+1, self.DoF))
            self.x_bar = self._predictTrajectories(xo, self.u_bar)

        x_bar_initial = self.x_bar.copy()
        u_bar_initial = self.u_bar.copy()

        # 0.2 Get ref, sdf map,
        r_bar_map = {}
        self.ree_bar = []
        self.rbase_bar = []
        for planner in planners:
            # get tracking points from planner, tracking points are tuples of desired position and velocity (p, v) 
            # for planners without velocity reference, none should be given (p, None) 
            if planner.ref_type == "path":
                p_bar, v_bar = planner.getTrackingPointArray((xo[:self.DoF], xo[self.DoF:]), self.N+1, self.dt)
            else:
                r_bar = [planner.getTrackingPoint(t + k * self.dt, (self.x_bar[k, :self.DoF], self.x_bar[k, self.DoF:]))
                            for k in range(self.N + 1)]
                p_bar = [r[0] for r in r_bar]
                v_bar = [r[1] for r in r_bar]
            contains_none = any(item is None for item in v_bar)
            velocity_ref_available = not contains_none
            
            acceptable_ref = True
            if planner.type == "EE":
                if planner.ref_data_type == "Vec3":
                    if planner.frame_id == "base_link":
                        r_bar_map["EEPos3BaseFrame"] = p_bar
                    else:
                        r_bar_map["EEPos3"] = p_bar
                        if velocity_ref_available:
                            # if planner generates velocity reference as well
                            r_bar_map["EEVel3"] = v_bar
                elif planner.ref_data_type == "SE3":
                    if planner.frame_id == "base_link":
                        r_bar_map["EEPoseBaseFrame"] = p_bar
                    else:
                        # assuming world frame
                        r_bar_map["EEPose"] = p_bar
                else:
                    acceptable_ref = False
            elif planner.type == "base":
                if planner.ref_data_type == "Vec2":
                    r_bar_map["BasePos2"] = p_bar
                    if velocity_ref_available:
                        r_bar_map["BaseVel2"] = v_bar
                elif planner.ref_data_type == "Vec3":
                    r_bar_map["BasePos3"] = p_bar
                    if velocity_ref_available:
                        r_bar_map["BaseVel3"] = v_bar
                elif planner.ref_data_type == "SE2":
                    r_bar_map["BasePoseSE2"] = p_bar
                    if velocity_ref_available:
                        r_bar_map["BaseVel3"] = v_bar
                else:
                    acceptable_ref = False

            if not acceptable_ref:
                self.py_logger.warning(f"unknown cost type {planner.ref_data_type}, planner {planner.name}")
            
            if planner.type == "EE":
                self.ree_bar = p_bar 
            elif planner.type == "base":
                self.rbase_bar = p_bar

        t1 = time.perf_counter()
        if map is not None and self.params["sdf_collision_avoidance_enabled"]:
            self.model_interface.sdf_map.update_map(*map)
        t2 = time.perf_counter()
        self.log["time_map_update"] = t2 - t1

        tp1 = time.perf_counter()
        curr_p_map_bar = []
        for key in self.log.keys():
            if "time" in key:
                self.log[key] = 0

        map_params = self.model_interface.sdf_map.get_params()
        for i in range(self.N+1):
            curr_p_map = self.p_struct(0)
            # curr_p_map["eps_Regularization"] = self.params["cost_params"]["Regularization"]["eps"]

            t1 = time.perf_counter()
            if self.params["sdf_collision_avoidance_enabled"]:
                # params = self.model_interface.sdf_map.get_params()
                curr_p_map["x_grid_sdf"] = map_params[0]
                curr_p_map["y_grid_sdf"] = map_params[1]
                if self.model_interface.sdf_map.dim == 3:
                    curr_p_map["z_grid_sdf"] = map_params[2]
                    curr_p_map["value_sdf"] = map_params[3]
                else:
                    curr_p_map["value_sdf"] = map_params[2]
            t2 = time.perf_counter()
            self.log["time_ocp_set_params_map"] += t2 - t1
            for name in self.collision_link_names:
                cbf_cst_type = False

                if name in self.model_interface.scene.collision_link_names["static_obstacles"]:
                    if self.params["collision_constraint_type"]["static_obstacles"] == "SignedDistanceConstraintCBF":
                        cbf_cst_type = True
                else:
                    if self.params["collision_constraint_type"][name] == "SignedDistanceConstraintCBF":
                        cbf_cst_type = True

                if cbf_cst_type:
                    p_name = "_".join(["gamma", name])
                    curr_p_map[p_name] = self.params["collision_cbf_gamma"][name]

            # set initial guess
            t1 = time.perf_counter()
            self.ocp_solver.set(i, 'x', x_bar_initial[i])
            if i < self.N:
                self.ocp_solver.set(i, 'u', u_bar_initial[i])
            if self.lam_bar is not None:
                self.ocp_solver.set(i, 'lam', self.lam_bar[i])

            t2 = time.perf_counter()
            self.log["time_ocp_set_params_set_x"] += t2 - t1
            # set parameters for tracking cost functions
            t1 = time.perf_counter()
            p_keys = self.p_struct.keys()
            for (name, r_bar) in r_bar_map.items():
                p_name_r = "_".join(["r", name])
                p_name_W = "_".join(["W", name])

                if p_name_r in p_keys:
                    # set reference
                    curr_p_map[p_name_r] = r_bar[i]

                    # Set weight matricies, assuming identity matrix with identical diagonal terms
                    if i == self.N:
                        curr_p_map[p_name_W] = np.diag(self.params["cost_params"][name]["P"])
                    else:
                        curr_p_map[p_name_W] = np.diag(self.params["cost_params"][name]["Qk"])
                else:
                    self.py_logger.warning(f"unknown p name {p_name_r}")
            
            curr_p_map["Qqa_ControlEffort"] = self.params["cost_params"]["Effort"]["Qqa"]
            curr_p_map["Qqb_ControlEffort"] = self.params["cost_params"]["Effort"]["Qqb"]
            curr_p_map["Qva_ControlEffort"] = self.params["cost_params"]["Effort"]["Qva"]
            curr_p_map["Qvb_ControlEffort"] = self.params["cost_params"]["Effort"]["Qvb"]
            curr_p_map["Qua_ControlEffort"] = self.params["cost_params"]["Effort"]["Qua"]
            curr_p_map["Qub_ControlEffort"] = self.params["cost_params"]["Effort"]["Qub"]

            t2 = time.perf_counter()
            self.log["time_ocp_set_params_tracking"] += t2 - t1

            t1 = time.perf_counter()
            self.ocp_solver.set(i, 'p', curr_p_map.cat.full().flatten())
            t2 = time.perf_counter()
            self.log["time_ocp_set_params_setp"] += t2 - t1
            curr_p_map_bar.append(curr_p_map)
        tp2 = time.perf_counter()
        self.log["time_ocp_set_params"] = tp2 - tp1

        t1 = time.perf_counter()
        self.ocp_solver.solve_for_x0(xo, fail_on_nonzero_status=False)
        t2 = time.perf_counter()
        self.log["time_ocp_solve"] = t2 - t1

        self.ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")
        self.log["solver_status"] = self.ocp_solver.status
        if self.ocp.solver_options.nlp_solver_type != "SQP_RTI":
            self.log["step_size"] = np.mean(self.ocp_solver.get_stats('alpha'))
        else:
            self.log["step_size"] = -1
        self.log["sqp_iter"] = self.ocp_solver.get_stats('sqp_iter')
        self.log["qp_iter"] = sum(self.ocp_solver.get_stats('qp_iter'))
        self.log["cost_final"] = self.ocp_solver.get_cost()

        if self.ocp_solver.status !=0:

            x_bar = []
            u_bar = []
            print(f"xo: {xo}")
            for i in range(self.N):
                print(f"stage {i}: x: {self.ocp_solver.get(i, 'x')}")
                x_bar.append(self.ocp_solver.get(i, 'x'))
                print(f"stage {i}: u: {self.ocp_solver.get(i, 'u')}")
                u_bar.append(self.ocp_solver.get(i, 'u'))
            
            x_bar.append(self.ocp_solver.get(self.N, 'x'))
                
            for i in range(self.N):
                print(f"stage {i}: lam: {self.ocp_solver.get(i, 'lam')}")
            
            for i in range(self.N):
                print(f"stage {i}: pi: {self.ocp_solver.get(i, 'pi')}")

            for i in range(self.N):
                print(f"stage {i}: sl: {self.ocp_solver.get(i, 'sl')}")
            for i in range(self.N):
                print(f"stage {i}: su: {self.ocp_solver.get(i, 'su')}")
                # v = self.evaluate_constraints(self.collisionCsts['sdf'], 
                #                                     x_bar, u_bar, curr_p_map_bar)
                # h = self.evaluate_sdf_h_fcn(self.collisionCsts['sdf'], 
                #                             x_bar, u_bar, curr_p_map_bar)
                # xdot = self.evaluate_sdf_xdot_fcn(self.collisionCsts['sdf'], 
                #                             x_bar, u_bar, curr_p_map_bar)
                # for i in range(self.N):
                #     print(f"stage {i}: t: {self.ocp_solver.get(i, 't')}")
                #     print(f"state {i}: sdf: {v[i]}")
                #     print(f"state {i}: h: {h[i]}")
                #     print(f"state {i}: xdot: {xdot[i]}")
            

            self.log["iter_snapshot"] = {"t": t,
                                         "xo": xo,
                                         "p_map_bar": [p.cat.full().flatten() for p in curr_p_map_bar],
                                         "x_bar_init": x_bar_initial,
                                         "u_bar_init": u_bar_initial,
                                         "x_bar": x_bar,
                                         "u_bar": u_bar,
                                         }

            # get iterate:
            solution = self.log["iter_snapshot"]

            lN = len(str(self.N+1))
            for i in range(self.N+1):
                i_string = f'{i:0{lN}d}'
                solution['x_'+i_string] = self.ocp_solver.get(i,'x')
                solution['u_'+i_string] = self.ocp_solver.get(i,'u')
                solution['z_'+i_string] = self.ocp_solver.get(i,'z')
                solution['lam_'+i_string] = self.ocp_solver.get(i,'lam')
                solution['t_'+i_string] = self.ocp_solver.get(i, 't')
                solution['sl_'+i_string] = self.ocp_solver.get(i, 'sl')
                solution['su_'+i_string] = self.ocp_solver.get(i, 'su')
                if i < self.N:
                    solution['pi_'+i_string] = self.ocp_solver.get(i,'pi')

            # for k in list(solution.keys()):
            #     if len(solution[k]) == 0:
            #         del solution[k]

            # self.ocp_solver.store_iterate(filename=str(self.output_dir / "iter_{:.2f}.json".format(t)))
            if self.params["acados"]["raise_exception_on_failure"]:
                raise Exception(f'acados acados_ocp_solver returned status {self.solver_status}')
        else:
            self.log["iter_snapshot"] = None

        # get solution
        self.u_prev = self.u_bar[0].copy()
        self.lam_bar = []
        for i in range(self.N):
            self.x_bar[i,:] = self.ocp_solver.get(i, "x")
            self.u_bar[i,:] = self.ocp_solver.get(i, "u")
            self.lam_bar.append(self.ocp_solver.get(i, "lam"))

        self.x_bar[self.N,:] = self.ocp_solver.get(self.N, "x")
        self.lam_bar.append(self.ocp_solver.get(self.N, "lam"))
        self.t_bar = t + np.arange(self.N) * self.dt

        self.v_cmd = self.x_bar[0][self.robot.DoF:].copy()

        # For rviz visualization
        t1 = time.perf_counter()

        self.ee_bar, self.base_bar = self._getEEBaseTrajectories(self.x_bar)
        self.sdf_bar["EE"] = self.model_interface.sdf_map.query_val(self.ee_bar[:, 0],self.ee_bar[:, 1],self.ee_bar[:, 2]).flatten()
        self.sdf_grad_bar["EE"] = self.model_interface.sdf_map.query_grad(self.ee_bar[:, 0],self.ee_bar[:, 1],self.ee_bar[:, 2]).reshape((3,-1))
        
        self.sdf_bar["base"] = self.model_interface.sdf_map.query_val(self.base_bar[:, 0], self.base_bar[:, 1], np.ones(self.N+1)*0.2)
        self.sdf_grad_bar["base"] = self.model_interface.sdf_map.query_grad(self.base_bar[:, 0], self.base_bar[:, 1], np.ones(self.N+1)*0.2).reshape((3,-1))

        for name in self.collision_link_names: 
            self.log["_".join([name, "constraint"])] = self.evaluate_constraints(self.collisionCsts[name], 
                                                                   self.x_bar, self.u_bar, curr_p_map_bar)
        #     # self.log["_".join([name, "constraint", "gradient"])] = self.evaluate_constraints_gradient(self.collisionCsts[name], 
        #     #                                                        self.x_bar, self.u_bar, curr_p_map_bar)
        # For data plotting
        # self.log["state_constraint"] = self.evaluate_constraints(self.stateCst, self.x_bar, self.u_bar, curr_p_map_bar)
        # self.log["control_constraint"] = self.evaluate_constraints(self.controlCst, self.x_bar, self.u_bar, curr_p_map_bar)
        self.log["ee_pos"] = self.ee_bar.copy()
        self.log["base_pos"] = self.base_bar.copy()
        self.log["ocp_param"] = [p.cat.full().flatten() for p in curr_p_map_bar]
        self.log["x_bar"] = self.x_bar.copy()
        self.log["u_bar"] = self.u_bar.copy()
        sdf_param = self.model_interface.sdf_map.get_params()
        for i, param in enumerate(sdf_param):
            self.log["_".join(["sdf", "param", str(i)])] = param
        t2 = time.perf_counter()
        self.log["time_ocp_overhead"] = t2 - t1
        return self.v_cmd, self.u_prev, self.u_bar.copy(), self.x_bar[:, 9:].copy()

    def _get_log(self):
        log = {"cost_final": 0,
               "step_size": 0,
               "sqp_iter": 0,
               "qp_iter": 0,
               "solver_status": 0,
               "time_map_update": 0,
               "time_ocp_set_params": 0,
               "time_ocp_solve": 0,
               "time_ocp_set_params_map" : 0,
               "time_ocp_set_params_set_x" : 0,
               "time_ocp_set_params_tracking" : 0,
               "time_ocp_set_params_setp" : 0,
               "state_constraint": 0,
               "control_constraint": 0,
               "x_bar": 0,
               "u_bar": 0,
               "lam_bar": 0,
               "ee_pos":0,
               "base_pos":0,
               "ocp_param":{},
               "iter_snapshot":{}
               }
        for name in self.collision_link_names:
            log["_".join([name, "constraint"])]= 0
            log["_".join([name, "constraint", "gradient"])]= 0

        for i in range(self.model_interface.sdf_map.dim+1):
            log["_".join(["sdf", "param", str(i)])] = 0
        return log
    
    def reset(self):
        super().reset()
        self.ocp_solver.reset()


if __name__ == "__main__":
    # robot mdl
    from mmseq_utils import parsing
    path_to_config = parse_ros_path({"package": "mmseq_run",
                           "path":"config/3d_collision_sdf.yaml"})
    config = parsing.load_config(path_to_config)

    controller = STMPC(config["controller"])
