import logging
from abc import abstractmethod
from pathlib import Path
from typing import Tuple

import casadi as cs
import mobile_manipulation_central as mm
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from numpy.typing import NDArray

import mm_control.MPCConstraints as MPCConstraints
from mm_control.MPCConstraints import ControlBoxConstraints, StateBoxConstraints
from mm_control.MPCCostFunctions import CostFunctions, SoftConstraintsRBFCostFunction
from mm_control.robot import CasadiModelInterface as ModelInterface
from mm_control.robot import MobileManipulator3D as MM
from mm_utils.casadi_struct import casadi_sym_struct
from mm_utils.parsing import parse_ros_path

INF = 1e5


class MPCBase:
    """Base class for Model Predictive Control"""

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
        self.tf = self.params["prediction_horizon"]
        self.N = int(self.tf / self.dt)
        self.QPsize = self.nx * (self.N + 1) + self.nu * self.N

        self.collision_link_names = (
            ["self"] if self.params["self_collision_avoidance_enabled"] else []
        )
        self.collision_link_names += (
            self.model_interface.scene.collision_link_names["static_obstacles"]
            if self.params["static_obstacles_collision_avoidance_enabled"]
            else []
        )

        self.collisionCsts = {}
        for name in self.collision_link_names:
            self.collisionCsts[name] = self._create_collision_constraint(name)

        self.collisionSoftCsts = {}
        for name, sd_cst in self.collisionCsts.items():
            self.collisionSoftCsts[name] = self._create_collision_soft_cost(
                name, sd_cst
            )

        self.stateCst = StateBoxConstraints(self.robot)
        self.controlCst = ControlBoxConstraints(self.robot)

        self.cost = []
        self.constraints = []

        self.x_bar = np.zeros((self.N + 1, self.nx))  # current best guess x0,...,xN
        self.x_bar[:, : self.DoF] = self.home
        self.u_bar = np.zeros((self.N, self.nu))  # current best guess u0,...,uN-1
        self.t_bar = None
        self.lam_bar = None  # inequality multipliers
        self.u_prev = np.zeros(self.nu)

        self.v_cmd = np.zeros(self.nx - self.DoF)

        self.py_logger = logging.getLogger("Controller")
        self.log = self._get_log()

        self.ree_bar = None
        self.rbase_bar = None
        self.ee_bar = None
        self.base_bar = None

        self.output_dir = Path(
            parse_ros_path({"package": "mm_control", "path": "acados_outputs"})
        )
        if not self.output_dir.exists():
            self.output_dir.mkdir()

    @abstractmethod
    def control(
        self,
        t: float,
        robot_states: Tuple[NDArray[np.float64], NDArray[np.float64]],
        references: dict,
    ):
        """
        :param t: current control time
        :param robot_states: (q, v) generalized coordinates and velocities
        :param references: Dictionary with reference trajectories from TaskManager:
            {
                "base_pose": array of shape (N+1, 3) or None,
                "base_velocity": array of shape (N+1, 3) or None,
                "ee_pose": array of shape (N+1, 6) or None,
                "ee_velocity": array of shape (N+1, 6) or None,
            }
        :return: v_bar, velocity trajectory, shape (N+1, nu)
        :return: u_bar, currently the best control inputs, aka, u_bar[0]
        """
        pass

    def _predictTrajectories(self, xo, u_bar):
        return MM.ssIntegrate(self.dt, xo, u_bar, self.ssSymMdl)

    def _getEEBaseTrajectories(self, x_bar):
        ee_bar = np.zeros((self.N + 1, 3))
        base_bar = np.zeros((self.N + 1, 3))
        for k in range(self.N + 1):
            base_bar[k] = x_bar[k, :3]
            fee_fcn = self.kinSymMdl[self.robot.tool_link_name]
            ee_pos, ee_orn = fee_fcn(x_bar[k, : self.DoF])
            ee_bar[k] = ee_pos.toarray().flatten()

        return ee_bar, base_bar

    def reset(self):
        self.x_bar = np.zeros((self.N + 1, self.nx))  # current best guess x0,...,xN
        self.u_bar = np.zeros((self.N, self.nu))  # current best guess u0,...,uN-1
        self.t_bar = None
        self.lam_bar = None

        self.v_cmd = np.zeros(self.nx - self.DoF)

    def evaluate_cost_function(
        self, cost_function: CostFunctions, x_bar, u_bar, nlp_p_map_bar
    ):
        cost_p_dict = cost_function.get_p_dict()
        cost_p_struct = casadi_sym_struct(cost_p_dict)
        cost_p_map = cost_p_struct(0)

        vals = []
        for k in range(self.N):
            nlp_p_map = nlp_p_map_bar[k]
            for key in cost_p_map.keys():
                cost_p_map[key] = nlp_p_map[key]
            v = cost_function.evaluate(
                x_bar[k], u_bar[k], cost_p_map.cat.full().flatten()
            )
            vals.append(v)
        return np.sum(vals)

    def evaluate_constraints(
        self, constraints: MPCConstraints.Constraint, x_bar, u_bar, nlp_p_map_bar
    ):
        cost_p_dict = constraints.get_p_dict()
        cost_p_struct = casadi_sym_struct(cost_p_dict)
        cost_p_map = cost_p_struct(0)

        vals = []
        for k in range(self.N + 1):
            nlp_p_map = nlp_p_map_bar[k]
            for key in cost_p_map.keys():
                cost_p_map[key] = nlp_p_map[key]
            if k < self.N:
                v = constraints.check(
                    x_bar[k], u_bar[k], cost_p_map.cat.full().flatten()
                )
            else:
                v = constraints.check(
                    x_bar[k], u_bar[k - 1], cost_p_map.cat.full().flatten()
                )

            vals.append(v)
        return vals

    def _construct(self, costs, constraints, num_terminal_cost, name="MM"):
        model, p_struct, p_map = self._setup_acados_model(costs, constraints, name)
        ocp = self._setup_acados_ocp(model, name)
        self._setup_costs(ocp, model, costs, num_terminal_cost)
        nsx, nsu, nsx_e, nsh, nsh_e, nsh_0 = self._setup_constraints(
            ocp, model, constraints
        )
        self._setup_slack_variables(ocp, nsx, nsu, nsh, nsx_e, nsh_e, nsh_0)

        ocp.constraints.x0 = self.x_bar[0]
        ocp.parameter_values = p_map.cat.full().flatten()
        self._configure_solver_options(ocp)
        ocp_solver = self._create_solver(ocp, name)

        return ocp, ocp_solver, p_struct

    def _setup_acados_model(self, costs, constraints, name):
        """Setup AcadosModel with dynamics and parameters."""
        model = AcadosModel()
        model.x = cs.MX.sym("x", self.nx)
        model.u = cs.MX.sym("u", self.nu)
        model.xdot = cs.MX.sym("xdot", self.nx)

        model.f_impl_expr = model.xdot - self.ssSymMdl["fmdl"](model.x, model.u)
        model.f_expl_expr = self.ssSymMdl["fmdl"](model.x, model.u)
        model.name = name

        # Get params from costs and constraints
        p_dict = {}
        for cost in costs:
            p_dict.update(cost.get_p_dict())
        for cst in constraints:
            p_dict.update(cst.get_p_dict())
        p_struct = casadi_sym_struct(p_dict)
        p_map = p_struct(0)
        model.p = p_struct.cat

        return model, p_struct, p_map

    def _setup_acados_ocp(self, model, name):
        """Setup AcadosOCP basic structure."""
        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.tf
        ocp.code_export_directory = str(self.output_dir / "c_generated_code")
        ocp.solver_options.ext_fun_compile_flags = "-O3"
        return ocp

    def _setup_costs(self, ocp, model, costs, num_terminal_cost):
        """Setup cost expressions for the OCP."""
        ocp.cost.cost_type = "EXTERNAL"
        cost_expr = []
        for cost in costs:
            Ji = cost.J_fcn(model.x, model.u, cost.p_sym)
            cost_expr.append(Ji)
        ocp.model.cost_expr_ext_cost = sum(cost_expr)

        custom_hess_expr = []
        if self.params["acados"]["use_custom_hess"]:
            for cost in costs:
                H_fcn = cost.get_custom_H_fcn()
                H_expr_i = H_fcn(model.x, model.u, cost.p_sym)
                custom_hess_expr.append(H_expr_i)
            ocp.model.cost_expr_ext_cost_custom_hess = sum(custom_hess_expr)

        if self.params["acados"]["use_terminal_cost"]:
            ocp.cost.cost_type_e = "EXTERNAL"
            cost_expr_e = sum(cost_expr[:num_terminal_cost])
            cost_expr_e = cs.substitute(cost_expr_e, model.u, [])
            model.cost_expr_ext_cost_e = cost_expr_e
            if self.params["acados"]["use_custom_hess"]:
                cost_hess_expr_e = sum(custom_hess_expr[:num_terminal_cost])
                cost_hess_expr_e = cs.substitute(cost_hess_expr_e, model.u, [])
                model.cost_expr_ext_cost_custom_hess_e = cost_hess_expr_e

    def _setup_constraints(self, ocp, model, constraints):
        """Setup all constraints (control, state, nonlinear) and return slack variable counts."""
        # Control input constraints
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

        # State constraints
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

        # Nonlinear constraints
        h_expr_list = []
        idxsh = []
        h_idx = 0
        for cst in constraints:
            h_expr_list.append(cst.g_fcn(model.x, model.u, cst.p_sym))
            if cst.slack_enabled and (
                self.params["acados"]["slack_enabled"]["h"]
                or self.params["acados"]["slack_enabled"]["h_0"]
                or self.params["acados"]["slack_enabled"]["h_e"]
            ):
                idxsh += [h_i for h_i in range(h_idx, h_idx + cst.ng)]
            h_idx += cst.ng

        nsh = len(idxsh) if self.params["acados"]["slack_enabled"]["h"] else 0
        nsh_e = len(idxsh) if self.params["acados"]["slack_enabled"]["h_e"] else 0
        nsh_0 = len(idxsh) if self.params["acados"]["slack_enabled"]["h_0"] else 0

        if len(h_expr_list) > 0:
            h_expr = cs.vertcat(*h_expr_list)
            h_expr_num = h_expr.shape[0]

            model.con_h_expr_0 = h_expr
            ocp.constraints.uh_0 = np.zeros(h_expr_num)
            ocp.constraints.lh_0 = -INF * np.ones(h_expr_num)

            model.con_h_expr = h_expr
            ocp.constraints.uh = np.zeros(h_expr_num)
            ocp.constraints.lh = -INF * np.ones(h_expr_num)

            model.con_h_expr_e = cs.substitute(h_expr, model.u, [])
            ocp.constraints.uh_e = np.zeros(h_expr_num)
            ocp.constraints.lh_e = -INF * np.ones(h_expr_num)

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
                ocp.constraints.lsh_e = np.zeros(nsh_e)
                ocp.constraints.ush_e = np.zeros(nsh_e)

        return nsx, nsu, nsx_e, nsh, nsh_e, nsh_0

    def _configure_solver_options(self, ocp):
        """Configure solver options from config."""
        for key, val in self.params["acados"]["ocp_solver_options"].items():
            attr = getattr(ocp.solver_options, key, None)
            if attr is not None:
                setattr(ocp.solver_options, key, val)
            else:
                self.py_logger.warning(
                    f"{key} not found in Acados solver options. Parameter is ignored."
                )

    def _create_solver(self, ocp, name):
        """Create and return AcadosOCPSolver."""
        json_file_name = str(self.output_dir / f"acados_ocp_{name}.json")
        if self.params["acados"]["cython"]["enabled"]:
            if self.params["acados"]["cython"]["recompile"]:
                AcadosOcpSolver.generate(ocp, json_file=json_file_name)
                AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
                return AcadosOcpSolver.create_cython_solver(json_file_name)
            else:
                return AcadosOcpSolver(
                    ocp, json_file=json_file_name, build=False, generate=False
                )
        else:
            return AcadosOcpSolver(ocp, json_file=json_file_name, build=True)

    def _create_collision_constraint(self, name):
        """Create a collision constraint for the given link name."""
        sd_fcn = self.model_interface.getSignedDistanceSymMdls(name)
        is_static = (
            name in self.model_interface.scene.collision_link_names["static_obstacles"]
        )

        if is_static:
            constraint_type_name = self.params["collision_constraint_type"][
                "static_obstacles"
            ]
            safety_margin = self.params["collision_safety_margin"]["static_obstacles"]
        else:
            constraint_type_name = self.params["collision_constraint_type"][name]
            safety_margin = self.params["collision_safety_margin"][name]

        collision_cst_type = getattr(MPCConstraints, constraint_type_name)
        return collision_cst_type(self.robot, sd_fcn, safety_margin, name)

    def _create_collision_soft_cost(self, name, sd_cst):
        """Create a soft collision cost for the given constraint."""
        expand = True
        is_static = (
            name in self.model_interface.scene.collision_link_names["static_obstacles"]
        )

        if is_static:
            mu = self.params["collision_soft"]["static_obstacles"]["mu"]
            zeta = self.params["collision_soft"]["static_obstacles"]["zeta"]
        else:
            mu = self.params["collision_soft"][name]["mu"]
            zeta = self.params["collision_soft"][name]["zeta"]

        return SoftConstraintsRBFCostFunction(
            mu, zeta, sd_cst, name + "CollisionSoftCst", expand=expand
        )

    def _setup_slack_variables(self, ocp, nsx, nsu, nsh, nsx_e, nsh_e, nsh_0):
        """Setup slack variables for the OCP."""
        z = self.params["cost_params"]["slack"]["z"]
        Z = self.params["cost_params"]["slack"]["Z"]

        ns = nsx + nsu + nsh
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

    def _get_log(self):
        return {}
