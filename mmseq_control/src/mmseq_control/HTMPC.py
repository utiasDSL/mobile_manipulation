from abc import ABC, abstractmethod
import time
import logging

import numpy as np
import casadi as cs
from cvxopt import solvers, matrix
from mosek import iparam, dparam
import mmseq_control.MPCConstraints as MPCConstraint
from mmseq_control.MPCCostFunctions import EEPos3CostFunction, BasePos2CostFunction, ControlEffortCostFunciton, SoftConstraintsRBFCostFunction, ControlEffortCostFuncitonNew, SumOfCostFunctions, EEPos3BaseFrameCostFunction
from mmseq_control.MPCConstraints import HierarchicalTrackingConstraint, HierarchicalTrackingConstraintCostValue, MotionConstraint, StateControlBoxConstraint, StateControlBoxConstraintNew, SignedDistanceCollisionConstraint, EEUpwardConstraint
from mmseq_control.robot import MobileManipulator3D as MM
from mmseq_control.robot import CasadiModelInterface as ModelInterface
from mmseq_utils.math import wrap_pi_array


class MPC():
    def __init__(self, config):
        self.model_interface = ModelInterface(config)
        self.robot = self.model_interface.robot
        self.ssSymMdl = self.robot.ssSymMdl
        self.kinSymMdl = self.robot.kinSymMdls
        self.nx = self.ssSymMdl["nx"]
        self.nu = self.ssSymMdl["nu"]
        self.DoF = self.robot.DoF

        self.params = config
        self.dt = self.params["dt"]
        self.N = int(self.params['prediction_horizon'] / self.params['dt'])
        self.QPsize = self.nx * (self.N + 1) + self.nu * self.N

        self.EEPos3Cost = EEPos3CostFunction(self.dt, self.N, self.robot, config["cost_params"]["EEPos3"])
        self.EEPos3BaseFrameCost = EEPos3BaseFrameCostFunction(self.dt, self.N, self.robot, config["cost_params"]["EEPos3"])
        self.BasePos2Cost = BasePos2CostFunction(self.dt, self.N, self.robot, config["cost_params"]["BasePos2"])
        if self.params["penalize_du"]:
            self.CtrlEffCost = ControlEffortCostFuncitonNew(self.dt, self.N, self.robot, config["cost_params"]["Effort"])
        else:
            self.CtrlEffCost = ControlEffortCostFunciton(self.dt, self.N, self.robot, config["cost_params"]["Effort"])

        self.rate = self.params["ctrl_rate"]

        self.x_bar = np.zeros((self.N + 1, self.nx))  # current best guess x0,...,xN
        self.u_bar = np.zeros((self.N, self.nu))  # current best guess u0,...,uN-1
        self.zopt = np.zeros(self.QPsize)  # current lineaization point
        self.u_prev = np.zeros(self.nu)
        self.xu_bar_init = False

        self.v_cmd = np.zeros(self.nx - self.DoF)
        if self.params["penalize_du"]:
            self.xuCst = StateControlBoxConstraintNew(self.dt, self.robot, self.N, tol=1e-5)
        else:
            self.xuCst = StateControlBoxConstraint(self.dt, self.robot, self.N, tol=1e-5)
        self.MotionCst = MotionConstraint(self.dt, self.N, self.robot)

        self.collision_link_names = ["self"] if self.params["self_collision_avoidance_enabled"] else []
        self.collision_link_names += self.model_interface.scene.collision_link_names["static_obstacles"] \
            if self.params["static_obstacles_collision_avoidance_enabled"] else []

        self.collisionCsts = {}
        for name in self.collision_link_names:
            sd_fcn = self.model_interface.getSignedDistanceSymMdls(name)
            sd_cst = SignedDistanceCollisionConstraint(self.robot, sd_fcn, self.dt, self.N,
                                                       self.params["collision_safety_margin"], name)
            self.collisionCsts[name] = sd_cst

        self.eeUpwardCst = EEUpwardConstraint(self.robot, self.params["ee_upward_deviation_angle_max"], self.dt, self.N)

        self.xuSoftCst = SoftConstraintsRBFCostFunction(self.params["xu_soft"]["mu"], self.params["xu_soft"]["zeta"], self.xuCst, "xuSoftCst")
        self.collisionSoftCsts = {name: SoftConstraintsRBFCostFunction(self.params["collision_soft"]["mu"],
                                                                       self.params["collision_soft"]["zeta"],
                                                                       sd_cst, name+"CollisionSoftCst")
                                                                       for name, sd_cst in self.collisionCsts.items()}
        self.eeUpwardSoftCst = SoftConstraintsRBFCostFunction(self.params["ee_upward_soft"]["mu"],
                                                              self.params["ee_upward_soft"]["zeta"],
                                                              self.eeUpwardCst, "eeUpwardSoftCst")
        self.x_bar_sym = cs.MX.sym('x_bar', self.nx, self.N + 1)
        self.u_bar_sym = cs.MX.sym('u_bar', self.nu, self.N)
        self.z_bar_sym = cs.vertcat(cs.vec(self.x_bar_sym), cs.vec(self.u_bar_sym))

        self.py_logger = logging.getLogger("Controller")

        # collecting data for visualization purpose
        self.ee_bar = np.zeros((self.N+1, 3))
        self.base_bar = np.zeros((self.N+1, 3))

        self.ree_bar = []
        self.rbase_bar = []

    @abstractmethod
    def control(self, t, robot_states, planners):
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
        self.zopt = np.zeros(self.QPsize)  # current lineaization point
        self.xu_bar_init = False

        self.v_cmd = np.zeros(self.nx - self.DoF)


class HTMPCSQP(MPC):
    def __init__(self, config):
        super().__init__(config)
        self.MotionCst = MotionConstraint(self.dt, self.N, self.robot, "DI Motion Model")
        self.hierarchy_cst_type = getattr(MPCConstraint, self.params["hierarchy_constraint_type"])

    def control(self, t, robot_states, planners):
        self.py_logger.debug("control time {}".format(t))
        self.curr_control_time = t
        q, v = robot_states
        q[2:9] = wrap_pi_array(q[2:9])
        xo = np.hstack((q, v))

        # 0.1 Get linearization point
        self.u_bar[:-1] = self.u_bar[1:]
        self.u_bar[-1] = 0
        self.x_bar = self._predictTrajectories(xo, self.u_bar)

        # 0.2 Get ref, cost_fcn, hierarchy constraint for each planner
        num_plans = len(planners)
        r_bars = []
        cost_fcns = []
        hier_csts = []

        for id, planner in enumerate(planners):
            r_bar = [planner.getTrackingPoint(t + k * self.dt, (self.x_bar[k, :self.DoF], self.x_bar[k, self.DoF:]))[0]
                     for k in range(self.N + 1)]
            r_bars.append([[np.array(r_bar)], [-self.u_prev]])

            if planner.type == "EE" and planner.ref_data_type == "Vec3" and planner.__class__.__name__ == "EESimplePlanner":
                cost_fcns.append([self.EEPos3Cost, self.CtrlEffCost])
            elif planner.type == "EE" and planner.ref_data_type == "Vec3" and planner.__class__.__name__ == "EESimplePlannerBaseFrame":
                cost_fcns.append([self.EEPos3BaseFrameCost, self.CtrlEffCost])
            elif planner.type == "base" and planner.ref_data_type == "Vec2":
                cost_fcns.append([self.BasePos2Cost, self.CtrlEffCost])
                # cost_fcns.append([self.BasePos2Cost])

            else:
                self.py_logger.warning("unknown cost type, planner # %d", id)

            if id < num_plans - 1:
                hier_csts.append(self.hierarchy_cst_type(cost_fcns[-1][0], planner.type))

            if planner.type == "EE":
                self.ree_bar = r_bar
            elif planner.type == "base":
                self.rbase_bar = r_bar

        xbar_opt, ubar_opt = self.solveHTMPC(xo, self.x_bar.copy(), self.u_bar.copy(), cost_fcns, hier_csts, r_bars)

        self.x_bar = xbar_opt.copy()
        self.u_bar = ubar_opt.copy()
        self.u_prev = self.u_bar[0].copy()
        self.v_cmd = self.x_bar[0][self.robot.DoF:].copy()
        # self.v_cmd = self.v_cmd + acc_cmd / self.rate
        # if not self.params["soft_cst"]:
        #     clamp_rate = 0.99
        #     self.v_cmd = np.where(self.v_cmd < self.robot.ub_x[self.robot.DoF:] * clamp_rate, self.v_cmd,
        #                           self.robot.ub_x[self.robot.DoF:] * clamp_rate)
        #     self.v_cmd = np.where(self.v_cmd > self.robot.lb_x[self.robot.DoF:] * clamp_rate, self.v_cmd,
        #                           self.robot.lb_x[self.robot.DoF:] * clamp_rate)
        self.ee_bar, self.base_bar = self._getEEBaseTrajectories(self.x_bar)

        return self.v_cmd, self.u_prev, self.u_bar.copy()

    def solveHTMPC(self, xo, xbar, ubar, cost_fcns, hier_csts, r_bars):
        """ HTMPC solver

        :param xbar: predicted state trajectory numpy.ndarray [N+1, nx]
        :param ubar: current best control trajectory numpy.ndarray [N, nu]
        :param cost_fcns: list of cost functions for each task in decreasing hierarchy, nested list
        :param hier_csts: list of hiearchical constraints for all task but the last one
        :param r_bars: list of reference trajectory for each cost fcn
        :return: xbar, ubar, the best solutions

        """
        task_num = len(cost_fcns)
        xbar_l = xbar.copy()
        ubar_l = ubar.copy()

        self.cost_iter = np.zeros((self.params["HT_MaxIntvl"], task_num, self.params["ST_MaxIntvl"] + 1))
        self.cost_final = np.zeros(task_num)

        self.step_size = np.zeros((self.params["HT_MaxIntvl"], task_num, self.params["ST_MaxIntvl"]))
        self.solver_status = np.zeros_like(self.step_size)

        self.tol_schedule = np.linspace(0.01, 0.01, self.params["HT_MaxIntvl"])

        for i in range(self.params["HT_MaxIntvl"]):
            e_bars = []
            J_bars = []
            if self.params["cst_tol_schedule_enabled"]:
                for cst in hier_csts:
                    cst.tol = self.tol_schedule[i]

            for task_id, cost_fcn in enumerate(cost_fcns):
                if self.params["soft_cst"]:
                    csts = {"eq": [self.MotionCst], "ineq": []}
                    csts_params = {"eq": [[xo]], "ineq": []}

                    # state control soft constraint function
                    if self.params["penalize_du"]:
                        xuSoft_param = [[self.u_prev]]
                    else:
                        xuSoft_param = [[]]

                    st_cost_fcn = cost_fcn + [self.xuSoftCst]
                    st_cost_fcn_params = r_bars[task_id] + xuSoft_param

                    for name, soft_cst in self.collisionSoftCsts.items():
                        st_cost_fcn += [soft_cst]
                        st_cost_fcn_params += [[]]

                    if self.params["ee_upward_constraint_enabled"]:
                        st_cost_fcn += [self.eeUpwardSoftCst]
                        st_cost_fcn_params += [[]]

                else:
                    csts = {"eq": [self.MotionCst], "ineq": [self.xuCst]}
                    csts_params = {"eq": [[xo]], "ineq": [[]]}

                    st_cost_fcn = cost_fcn
                    st_cost_fcn_params = r_bars[task_id]

                if task_id > 0:
                    csts["ineq"] += hier_csts[:task_id]
                    for prev_task_id in range(task_id):
                        if self.params["hierarchy_constraint_type"] == "HierarchicalTrackingConstraint":
                            csts_params["ineq"].append([r_bars[prev_task_id][0][0].T, e_bars[prev_task_id]])
                        elif self.params["hierarchy_constraint_type"] == "HierarchicalTrackingConstraintCostValue":
                            csts_params["ineq"].append([r_bars[prev_task_id][0][0].T, J_bars[prev_task_id]])




                # t0 = time.perf_counter()
                xbar_lopt, ubar_lopt, status = self.solveSTMPCCasadi(xo, xbar_l.copy(), ubar_l.copy(), st_cost_fcn,
                                                                     st_cost_fcn_params, csts, csts_params, i, task_id)

                # t1 = time.perf_counter()
                # print("STMPC Time: {}".format(t1 - t0))
                if task_id < task_num - 1:
                    e_bars_l = cost_fcn[0].e_bar_fcn(xbar_lopt.T, ubar_lopt.T, r_bars[task_id][0][0].T)
                    e_bars.append(e_bars_l.toarray().flatten())

                    J_bar_l = cost_fcn[0].J_bar_fcn(xbar_lopt.T, ubar_lopt.T, r_bars[task_id][0][0].T)
                    J_bars.append(J_bar_l)

                xbar_l = xbar_lopt.copy()
                ubar_l = ubar_lopt.copy()

        for task_id, cost_fcn in enumerate(cost_fcns):
            self.cost_final[task_id] = self._eval_cost_functions(cost_fcn, xbar_l, ubar_l, r_bars[task_id])
            # if self.params["soft_cst"]:
            #     self.cost_final[task_id] += self.xuSoftCst.evaluate(xbar_l, ubar_l, *xuSoft_param[0])
        return xbar_l, ubar_l


    def solveSTMPCCasadi(self, xo, xbar, ubar, cost_fcn, cost_fcn_params, csts, csts_params, ht_iter=0, task_id=0):
        """

        :param xo:
        :param xbar:
        :param ubar:
        :param cost_fcn: a list of cost functions to be added together
        :param cost_fcn_params: a list of params to be passed to each cost function
        :param csts: a dictionary, equality and inequality constraint list
        :param csts_params: parameters for each constraint
        :param ht_iter: current HT-MPC Solver iteration
        :param task_id: current st task id
        :return: xbar_opt, ubar_opt
        """
        xbar_i = xbar.copy()
        ubar_i = ubar.copy()
        self.cost_iter[ht_iter, task_id, 0] = self._eval_cost_functions(cost_fcn, xbar_i, ubar_i, cost_fcn_params)

        for i in range(self.params["ST_MaxIntvl"]):
            tp0 = time.perf_counter()
            # t0 = time.perf_counter()
            # Cost Function
            H = cs.DM.zeros((self.QPsize, self.QPsize))
            H += cs.DM.eye(self.QPsize) * 1e-6
            g = cs.DM.zeros(self.QPsize)
            for id, f in enumerate(cost_fcn):
                Hi, gi = f.quad(xbar_i, ubar_i, *cost_fcn_params[id])
                H += Hi
                g += gi
            # t1 = time.perf_counter()
            # print("Cost Function Prep Time:{}".format(t1 - t0))

            # Equality Constraints
            # t0 = time.perf_counter()

            A = cs.DM.zeros((0, self.QPsize))
            b = cs.DM.zeros(0)

            for id, cst in enumerate(csts["eq"]):
                Ai, bi = cst.linearize(xbar_i, ubar_i, *csts_params["eq"][id])
                A = cs.vertcat(A, Ai)
                b = cs.vertcat(b, bi)
            # t1 = time.perf_counter()
            # print("Eq Constraint Prep Time:{}".format(t1 - t0))

            # Inequality Constraints (without state bound)
            # t0 = time.perf_counter()
            C = cs.DM.zeros((0, self.QPsize))
            d = cs.DM.zeros(0)
            if self.params['soft_cst']:
                for id, cst in enumerate(csts["ineq"]):
                    Ci, di = cst.linearize(xbar_i, ubar_i, *csts_params["ineq"][id])
                    C = cs.vertcat(C, Ci)
                    d = cs.vertcat(d, di)
            else:
                for id, cst in enumerate(csts["ineq"][1:]):
                    Ci, di = cst.linearize(xbar_i, ubar_i, *csts_params["ineq"][id + 1])
                    C = cs.vertcat(C, Ci)
                    d = cs.vertcat(d, di)

            # C_scaled, d_scaled = self.scaleConstraints(C.toarray(), d.toarray().flatten())

            # State Bound
            if not self.params["soft_cst"]:
                _, bx = self.xuCst.linearize(xbar_i, ubar_i)

            Ac = cs.vertcat(A, C)
            uba = cs.vertcat(-b, -d)
            lba = cs.vertcat(-b, -cs.DM.inf(d.shape[0]))


            qp = {}
            qp['h'] = H.sparsity()
            qp['a'] = Ac.sparsity()
            opts = {"error_on_fail": True,
                    "gurobi": {"OutputFlag": 0, "LogToConsole": 0, "Presolve": 1, "BarConvTol": 1e-8,
                               "OptimalityTol": 1e-6}}
            S = cs.conic('S', 'gurobi', qp, opts)

            tp1 = time.perf_counter()
            self.py_logger.log(5, "QP prep time:{}".format(tp1 - tp0))

            t0 = time.perf_counter()
            try:
                if self.params["soft_cst"]:
                    results = S(h=H, g=g, a=Ac, uba=uba, lba=lba)
                else:
                    results = S(h=H, g=g, a=Ac, uba=uba, lba=lba, lbx=bx[self.QPsize:], ubx=-bx[:self.QPsize])

                results['status_val'] = 1
                results['status'] = 'optimal'
            except RuntimeError:
                if not self.xuCst.check(xbar_i, ubar_i)[0]:
                    results = {}
                    results['status'] = 'infeasible'
                    results['status_val'] = -1
                    results['x'] = np.zeros(self.QPsize)
                else:
                    results = {}
                    results['status'] = 'unknown'
                    results['status_val'] = -10
                    results['x'] = np.zeros(self.QPsize)

            t1 = time.perf_counter()
            self.py_logger.log(5, "QP time:{}".format(t1 - t0))

            dzopt = np.array(results['x']).squeeze()

            dubar = dzopt[self.nx * (self.N + 1):]
            t0 = time.perf_counter()
            if results['status'] == 'optimal':
                linesearch_step_opt, _ = self.lineSearchNew(xo, ubar_i, dubar, cost_fcn, cost_fcn_params, csts,
                                                            csts_params, dzopt)

                ubar_opt_i = ubar_i + linesearch_step_opt * dubar.reshape((self.N, self.nu))
                xbar_opt_i = self._predictTrajectories(xo, ubar_opt_i)
                xbar_i, ubar_i = xbar_opt_i.copy(), ubar_opt_i.copy()
            else:
                linesearch_step_opt = 0.

            self.cost_iter[ht_iter, task_id, i + 1] = self._eval_cost_functions(cost_fcn, xbar_i, ubar_i,
                                                                                cost_fcn_params)
            self.step_size[ht_iter, task_id, i] = linesearch_step_opt
            self.solver_status[ht_iter, task_id, i] = results['status_val']
            t1 = time.perf_counter()
            self.py_logger.log(5, "Line Search time:{}".format(t1 - t0))

        return xbar_i, ubar_i, results['status']

    def scaleConstraints(self, G, h):
        largeval_idx = np.where(np.logical_or(h > 1e-2, h < -1e-2))
        # if np.any(harray[largeval_idx] < 0.):
        #     print(1)
        G[largeval_idx] = G[largeval_idx] / np.abs(h[largeval_idx]).reshape((len(largeval_idx[0]), 1))
        h[largeval_idx] = 1. * (h[largeval_idx] > 0) - 1. * (h[largeval_idx] < 0)

        smallval_idx = np.where(np.logical_and(h < 1e-2, h > -1e-2))
        h[smallval_idx] = 0.

        # smallval_G_idx = np.where(np.abs(G) < 1e-10)
        # G[smallval_G_idx] = 0.

        return G.copy(), h.copy()

    def scaleConstraintsNew(self, G, h):
        smallval_idx = np.where(np.abs(h) < 1e-6)
        h[smallval_idx] = 0.

        smallval_G_idx = np.where(np.abs(G) < 1e-10)
        G[smallval_G_idx] = 0.

        return G.copy(), h.copy()

    def checkLinConstraintsFeas(self, G, h, A, b, dz, tol=1e-2):
        feas = True
        ineq_vio = G @ dz + h
        ineq_vio_indx = np.where(ineq_vio > tol)
        if len(ineq_vio_indx[0] > 0):
            self.py_logger.debug("Linearized Inequality Constraints # %d infeasible!", ineq_vio_indx[0])
            feas = False

        eq_vio = np.abs(A @ dz + b)
        eq_vio_indx = np.where(eq_vio > tol)

        if len(eq_vio_indx[0] > 0):
            self.py_logger.debug("Linearized Equality Constraints #{} infeasible!".format(eq_vio_indx[0]))
            feas = False

        return feas

    def lineSearchNew(self, xo, ubar, dubar, cost_fcn, cost_fcn_params, csts, csts_params, dzopt):
        # scale factor starts at 1
        t = 1

        beta = self.params["beta"]  # backtracking coefficient
        alpha = self.params["alpha"]  # discount in decrement
        MAX_ITER = 10

        dubar_rp = dubar.reshape((self.N, self.nu))
        xbar = self._predictTrajectories(xo, ubar)
        # print(xbar[:, 9])
        # print(ubar[:, 0])
        # print(xo)
        for k in range(MAX_ITER):
            ubar_new = ubar + t * dubar_rp
            xbar_new = self._predictTrajectories(xo, ubar_new)

            feas = True
            for cst_id, cst in enumerate(csts["ineq"]):
                feas_i = cst.check(xbar_new, ubar_new, *csts_params["ineq"][cst_id])[0]
                if not feas_i:
                    feas = False
                    # print(xbar_new[:, 9])
                    self.py_logger.debug(
                        "Controller: line search step {} not feasible. Violating constraint {}".format(t, cst.name))
                    break

            if feas:
                J_xp = 0
                J_xp_lin = 0
                for fid, f in enumerate(cost_fcn):
                    J_xp += f.evaluate(xbar_new, ubar_new, *cost_fcn_params[fid])
                    _, g = f.quad(xbar, ubar, *cost_fcn_params[fid])
                    J_xp_lin += f.evaluate(xbar, ubar, *cost_fcn_params[fid]) + alpha * t * g.T @ dzopt
                if J_xp < J_xp_lin:
                    return t, J_xp
                else:
                    self.py_logger.debug("Controller: line search step acceptance condition not met {}.".format(J_xp_lin - J_xp))

            t = t * beta

        J = 0
        for fid, f in enumerate(cost_fcn):
            J += f.evaluate(xbar, ubar, *cost_fcn_params[fid])
        return 0, J

    def _eval_cost_functions(self, cost_fcn, xbar, ubar, cost_fcn_params):
        J = 0
        for id, f in enumerate(cost_fcn):
            Ji = f.evaluate(xbar, ubar, *cost_fcn_params[id])
            J += Ji
            self.py_logger.log(4, f.name+" value:{}".format(Ji))
        return J


class HTMPCLex(HTMPCSQP):
    def __init__(self, config):
        super().__init__(config)

    def solveHTMPC(self, xo, xbar, ubar, cost_fcns, hier_csts, r_bars):
        """ HTMPC Lex IPOPT solver

                :param xbar: predicted state trajectory numpy.ndarray [N+1, nx]
                :param ubar: current best control trajectory numpy.ndarray [N, nu]
                :param cost_fcns: list of cost functions for each task in decreasing hierarchy, nested list
                :param hier_csts: list of hiearchical constraints for all task but the last one
                :param r_bars: list of reference trajectory for each cost fcn
                :return: xbar, ubar, the best solutions

                """
        task_num = len(cost_fcns)
        xbar_l = xbar.copy()
        ubar_l = ubar.copy()

        self.cost_iter = np.zeros((self.params["HT_MaxIntvl"], task_num, 2))

        self.cost_final = np.zeros(task_num)
        self.solver_status = None
        self.step_size = None

        e_bars = []
        for i in range(self.params["HT_MaxIntvl"]):
            for task_id, cost_fcn in enumerate(cost_fcns):
                csts = {"eq": [self.MotionCst], "ineq": []}
                csts_params = {"eq": [[xo]], "ineq": []}

                if task_id > 0:
                    csts["ineq"] += hier_csts[:task_id]
                    for prev_task_id in range(task_id):
                        csts_params["ineq"].append([r_bars[prev_task_id][0][0].T, e_bars[prev_task_id]])

                xbar_lopt, ubar_lopt, status = self.solveSTMPC(xo, xbar_l.copy(), ubar_l.copy(), cost_fcn, r_bars[task_id],
                                                               csts, csts_params, ht_iter=i, task_id=task_id)

                e_bars_l = cost_fcn[0].e_bar_fcn(xbar_lopt.T, ubar_lopt.T, r_bars[task_id][0][0].T)
                e_bars.append(e_bars_l.toarray().flatten())
                xbar_l = xbar_lopt.copy()
                ubar_l = ubar_lopt.copy()

        for task_id, cost_fcn in enumerate(cost_fcns):
            self.cost_final[task_id] = self._eval_cost_functions(cost_fcn, xbar_l, ubar_l, r_bars[task_id])

        return xbar_l, ubar_l

    def solveSTMPC(self, xo, xbar, ubar, cost_fcn, cost_fcn_params, csts, csts_params, ht_iter=0, task_id=0):
        self.cost_iter[ht_iter, task_id, 0] = self._eval_cost_functions(cost_fcn, xbar, ubar, cost_fcn_params)
        # f (x)
        f_eqn = cs.DM.zeros()
        for cost_id, cost in enumerate(cost_fcn):
            if cost.__class__.__name__ != "ControlEffortCostFunciton":
                J_eqn = cost.J_fcn(self.x_bar_sym, self.u_bar_sym, cost_fcn_params[cost_id][0].T)
            else:
                Jxu_eqn = cost.xu_cost.J_fcn(self.x_bar_sym, self.u_bar_sym, np.zeros(self.QPsize))
                Jdu_eqn = cost.du_cost.J_fcn(self.x_bar_sym, self.u_bar_sym,
                                             np.hstack((cost_fcn_params[cost_id][0], np.zeros(self.nu * (self.N-1)))))
                J_eqn = Jxu_eqn + Jdu_eqn

            f_eqn += J_eqn

        # Equality Constraint
        glb = cs.DM.zeros(0)
        gub = cs.DM.zeros(0)
        g_eqn = cs.MX.sym('g', 0)
        for cst_id, cst in enumerate(csts["eq"]):
            h_eqn = cst.h_fcn(self.x_bar_sym, self.u_bar_sym, *csts_params["eq"][cst_id])
            g_eqn = cs.vertcat(g_eqn, h_eqn)
            p, _ = h_eqn.shape
            glb = cs.vertcat(glb, cs.DM.zeros(p))
            gub = cs.vertcat(gub, cs.DM.zeros(p))

        # Inequality Constraint
        for cst_id, cst in enumerate(csts["ineq"]):
            gi_eqn = cst.g_fcn(self.x_bar_sym, self.u_bar_sym, *csts_params["ineq"][cst_id])
            g_eqn = cs.vertcat(g_eqn, gi_eqn)
            p, _ = gi_eqn.shape
            glb = cs.vertcat(glb, cs.DM.ones(p) * -cs.inf)
            gub = cs.vertcat(gub, cs.DM.zeros(p))

        nlp = {'x': self.z_bar_sym, 'f': f_eqn, 'g': g_eqn}
        opts = {"ipopt": {'linear_solver': "ma86", 'print_level': 0, "warm_start_init_point":'yes'}}  # , 'constr_viol_tol': 1e-5, 'tol': 1e-5}}
        S = cs.nlpsol('S', 'ipopt', nlp, opts)

        # Solve NLP

        u_init_base = np.ones((self.N, 2)) * 2
        u_init = np.hstack((u_init_base, np.zeros((self.N, self.nu - 2))))
        x_init = self._predictTrajectories(xo, u_init)
        x_init = np.hstack((x_init.flatten(), u_init.flatten()))
        res = S(x0=x_init, lbx=self.xuCst.lb, ubx=self.xuCst.ub, lbg=glb, ubg=gub)
        z_opt = res['x']

        xbar_new = z_opt[:self.nx * (self.N + 1)].reshape((self.nx, self.N + 1))
        ubar_new = z_opt[self.nx * (self.N + 1):].reshape((self.nu, self.N))

        xbar_new = xbar_new.T.toarray()
        ubar_new = ubar_new.T.toarray()

        self.cost_iter[ht_iter, task_id, 1] = self._eval_cost_functions(cost_fcn, xbar_new, ubar_new, cost_fcn_params)

        return xbar_new, ubar_new, 0
