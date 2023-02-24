from abc import ABC, abstractmethod
import time
import logging

import numpy as np
import casadi as cs
from cvxopt import solvers, matrix
from mosek import iparam, dparam

from mmseq_control.MPCCostFunctions import EEPos3CostFunction, BasePos2CostFunction, ControlEffortCostFunciton
from mmseq_control.MPCConstraints import HierarchicalTrackingConstraint, MotionConstraint, StateControlBoxConstraint
from mmseq_control.robot import MobileManipulator3D as MM



class MPC():
    def __init__(self, config):
        self.robot = MM(config)
        self.ssSymMdl = self.robot.ssSymMdl
        self.nx = self.ssSymMdl["nx"]
        self.nu = self.ssSymMdl["nu"]
        self.DoF = self.robot.DoF

        self.params = config
        self.dt = self.params["dt"]
        self.N = int(self.params['prediction_horizon']/self.params['dt'])
        self.QPsize = self.nx * (self.N + 1) + self.nu * self.N

        self.EEPos3Cost = EEPos3CostFunction(self.dt, self.N, self.robot, config["cost_params"]["EEPos3"])
        self.BasePos2Cost = BasePos2CostFunction(self.dt, self.N, self.robot, config["cost_params"]["BasePos2"])
        self.CtrlEffCost = ControlEffortCostFunciton(self.dt, self.N, self.robot, config["cost_params"]["Effort"])

        self.sim_dt = self.params["sim_dt"]

        self.x_bar = np.zeros((self.N + 1, self.nx))  # current best guess x0,...,xN
        self.u_bar = np.zeros((self.N, self.nu))  # current best guess u0,...,uN-1
        self.zopt = np.zeros(self.QPsize)  # current lineaization point
        self.xu_bar_init = False

        self.v_cmd = np.zeros(self.nx - self.DoF)

        self.xuCst = StateControlBoxConstraint(self.dt, self.robot, self.N)
        self.MotionCst = MotionConstraint(self.dt, self.N, self.robot)

        self.x_bar_sym = cs.MX.sym('x_bar', self.nx, self.N + 1)
        self.u_bar_sym = cs.MX.sym('u_bar', self.nu, self.N)
        self.z_bar_sym = cs.vertcat(cs.vec(self.x_bar_sym), cs.vec(self.u_bar_sym))

        self.py_logger = logging.getLogger("Controller")

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

    def reset(self):
        self.x_bar = np.zeros((self.N + 1, self.nx))  # current best guess x0,...,xN
        self.u_bar = np.zeros((self.N, self.nu))  # current best guess u0,...,uN-1
        self.zopt = np.zeros(self.QPsize)  # current lineaization point
        self.xu_bar_init = False

        self.v_cmd = np.zeros(self.nx - self.DoF)


class HTMPC(MPC):
    def __init__(self, config):
        super().__init__(config)

        self.EEPos3HCst = HierarchicalTrackingConstraint(self.EEPos3Cost, "EEPos3")
        self.BasePos2HCst = HierarchicalTrackingConstraint(self.BasePos2Cost, "BasePos2")
        self.MotionCst = MotionConstraint(self.dt, self.N, self.robot, "DI Motion Model")

    def control(self, t, robot_states, planners):
        q, v = robot_states
        xo = np.hstack((q, v))

        # 0.1 Get linearization point
        self.u_bar[:-1] = self.u_bar[1:]
        self.x_bar = self._predictTrajectories(xo, self.u_bar)

        # 0.2 Get ref, cost_fcn, hierarchy constraint for each planner
        num_plans = len(planners)
        r_bars = []
        cost_fcns = []
        hier_csts = []

        for id, planner in enumerate(planners):
            r_bar = [planner.getTrackingPoint(t+k*self.dt, (self.x_bar[k, :self.DoF], self.x_bar[k, self.DoF:]))[0] for k in range(self.N+1)]
            r_bars.append([np.array(r_bar)])
            if planner.cost_type == "EEPos3":
                cost_fcns.append([self.EEPos3Cost])
            elif planner.cost_type == "BasePos2":
                cost_fcns.append([self.BasePos2Cost])
            else:
                self.py_logger.warning("unknown cost type, planner # %d", id)

            if id < num_plans - 1:
                hier_csts.append(HierarchicalTrackingConstraint(cost_fcns[-1][0], planner.type))
            else:
                r_bars[-1].append(np.zeros(self.CtrlEffCost.nr))
                cost_fcns[-1].append(self.CtrlEffCost)


        xbar_opt, ubar_opt = self.solveHTMPC(xo, self.x_bar.copy(), self.u_bar.copy(), cost_fcns, hier_csts, r_bars)

        self.x_bar = xbar_opt.copy()
        self.u_bar = ubar_opt.copy()
        acc_cmd = self.u_bar[0]
        self.v_cmd = self.v_cmd + acc_cmd * self.dt
        return v + acc_cmd * self.dt, acc_cmd

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

        self.cost_iter = np.zeros((self.params["HT_MaxIntvl"], task_num, self.params["ST_MaxIntvl"]+1))
        self.cost_final = np.zeros(task_num)
        self.stmpc_run_time = np.zeros((self.params["HT_MaxIntvl"], task_num))

        self.step_size = np.zeros((self.params["HT_MaxIntvl"], task_num, self.params["ST_MaxIntvl"]))
        self.solver_status = np.zeros_like(self.step_size)

        self.tol_schedule = np.linspace(0.1, 0.01, self.params["HT_MaxIntvl"])

        for i in range(self.params["HT_MaxIntvl"]):
            e_bars = []
            if self.params["type"] == "SQP_TOL_SCHEDULE":
                for cst in hier_csts:
                    cst.tol = self.tol_schedule[i]

            for task_id, cost_fcn in enumerate(cost_fcns):
                csts = {"eq": [self.MotionCst], "ineq": [self.xuCst]}
                csts_params = {"eq": [[xo]], "ineq": [[]]}

                if task_id > 0:
                    csts["ineq"] += hier_csts[:task_id]
                    for prev_task_id in range(task_id):
                        csts_params["ineq"].append([r_bars[prev_task_id][0].T, e_bars[prev_task_id]])

                t0 = time.perf_counter()
                xbar_lopt, ubar_lopt, status = self.solveSTMPCCasadi(xo, xbar_l.copy(), ubar_l.copy(), cost_fcn, r_bars[task_id], csts, csts_params, i, task_id)
                t1 = time.perf_counter()
                # print("STMPC Time: {}".format(t1 - t0))
                self.stmpc_run_time[i, task_id] = t1 - t0

                e_bars_l = cost_fcn[0].e_bar_fcn(xbar_lopt.T, ubar_lopt.T, r_bars[task_id][0].T)
                e_bars.append(e_bars_l.toarray().flatten())
                xbar_l = xbar_lopt.copy()
                ubar_l = ubar_lopt.copy()

        for task_id, cost_fcn in enumerate(cost_fcns):
            self.cost_final[task_id] = self._eval_cost_functions(cost_fcn, xbar_l, ubar_l, r_bars[task_id])

        return xbar_l, ubar_l

    def solveSTMPC(self, xo, xbar, ubar, cost_fcn, cost_fcn_params, csts, csts_params, ht_iter=0, task_id=0):
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
            # tp0 = time.perf_counter()
            # t0 = time.perf_counter()
            # Cost Function
            H = cs.DM.zeros((self.QPsize, self.QPsize))
            H += cs.DM.eye(self.QPsize) * 1e-6
            g = cs.DM.zeros(self.QPsize)
            for id, f in enumerate(cost_fcn):
                Hi, gi = f.quad(xbar_i, ubar_i, cost_fcn_params[id])
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

            # Inequality Constraints
            # t0 = time.perf_counter()
            C = cs.DM.zeros((0, self.QPsize))
            d = cs.DM.zeros(0)
            for id, cst in enumerate(csts["ineq"]):
                Ci, di = cst.linearize(xbar_i, ubar_i, *csts_params["ineq"][id])
                C = cs.vertcat(C, Ci)
                d = cs.vertcat(d, di)

            C_scaled, d_scaled = self.scaleConstraints(C.toarray(), d.toarray().flatten())
            # t1 = time.perf_counter()
            # print("InEq Constraint Prep Time:{}".format(t1 - t0))

            # tp1 = time.perf_counter()
            # print("QP Prep Time:{}".format(tp1 - tp0))
            # Solve QP
            P = matrix(H.toarray())
            q = matrix(g.toarray())
            G = matrix(C_scaled)
            h = matrix(-d_scaled)
            A = matrix(A.toarray())
            b = matrix(-b.toarray())
            solvers.options['mosek'] = {iparam.log: 0, iparam.max_num_warnings: 0}
                                        # dparam.intpnt_qo_tol_pfeas: 1e-5,
                                        # dparam.intpnt_qo_tol_dfeas: 1e-5,
                                        # dparam.intpnt_qo_tol_rel_gap: 1e-5,
                                        # dparam.intpnt_qo_tol_infeas: 1e-5}
            initval = {'x': matrix(np.zeros(self.QPsize))}
            # t0 = time.perf_counter()
            results = solvers.qp(P, q, G, h, A, b, initvals=initval, solver='mosek')
            # t1 = time.perf_counter()
            # print("QP time:{}".format(t1 - t0))
            if results['status'] == 'optimal':
                results['status_val'] = 1
            elif results['status'] == 'unknown':
                results['status_val'] = 0
                results['x'] = np.zeros(g.size)
            else:
                results['status_val'] = -10
                results['x'] = np.zeros(g.size)

            dzopt = np.array(results['x']).squeeze()
            dubar = dzopt[self.nx * (self.N + 1):]
            linesearch_step_opt = 1.
            # t0 = time.perf_counter()
            if results['status'] == 'optimal':
                linesearch_step_opt, J_opt = self.lineSearch(xo, ubar_i, dubar, cost_fcn, cost_fcn_params, csts, csts_params)
            else:
                linesearch_step_opt = 0

            ubar_opt_i = ubar_i + linesearch_step_opt*dubar.reshape((self.N, self.nu))
            xbar_opt_i = self._predictTrajectories(xo, ubar_opt_i)
            xbar_i, ubar_i = xbar_opt_i.copy(), ubar_opt_i.copy()


            self.cost_iter[ht_iter, task_id, i+1] = self._eval_cost_functions(cost_fcn, xbar_i, ubar_i, cost_fcn_params)
            self.step_size[ht_iter, task_id, i] = linesearch_step_opt
            self.solver_status[ht_iter, task_id, i] = results['status_val']
            # t1 = time.perf_counter()
            # print("Line Search time:{}".format(t1 - t0))

        return xbar_i, ubar_i, results['status']

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
            # tp0 = time.perf_counter()
            # t0 = time.perf_counter()
            # Cost Function
            H = cs.DM.zeros((self.QPsize, self.QPsize))
            H += cs.DM.eye(self.QPsize) * 1e-6
            g = cs.DM.zeros(self.QPsize)
            for id, f in enumerate(cost_fcn):
                Hi, gi = f.quad(xbar_i, ubar_i, cost_fcn_params[id])
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
            for id, cst in enumerate(csts["ineq"][1:]):
                Ci, di = cst.linearize(xbar_i, ubar_i, *csts_params["ineq"][id+1])
                C = cs.vertcat(C, Ci)
                d = cs.vertcat(d, di)

            # C_scaled, d_scaled = self.scaleConstraints(C.toarray(), d.toarray().flatten())

            # State Bound
            _, bx = self.xuCst.linearize(xbar_i, ubar_i)

            Ac = cs.vertcat(A, C)
            uba = cs.vertcat(-b, -d)
            lba = cs.vertcat(-b, -cs.DM.inf(d.shape[0]))

            qp = {}
            qp['h'] = H.sparsity()
            qp['a'] = Ac.sparsity()
            opts= {"error_on_fail": False}
            S = cs.conic('S', 'gurobi', qp, opts)

            # H = H.toarray()
            # H = np.where(H < 1e-8, 0, H)

            t0 = time.perf_counter()
            results = S(h=H, g=g, a=Ac, uba=uba, lba=lba, lbx=bx[self.QPsize:], ubx=-bx[:self.QPsize], x0=cs.DM.zeros(self.QPsize))

            t1 = time.perf_counter()
            print("QP time:{}".format(t1 - t0))

            results['status_val'] = 1
            results['status'] = 'optimal'

            dzopt = np.array(results['x']).squeeze()
            dubar = dzopt[self.nx * (self.N + 1):]
            linesearch_step_opt = 1.
            # t0 = time.perf_counter()
            if results['status'] == 'optimal':
                linesearch_step_opt, J_opt = self.lineSearch(xo, ubar_i, dubar, cost_fcn, cost_fcn_params, csts, csts_params)
            else:
                linesearch_step_opt = 0

            ubar_opt_i = ubar_i + linesearch_step_opt*dubar.reshape((self.N, self.nu))
            xbar_opt_i = self._predictTrajectories(xo, ubar_opt_i)
            xbar_i, ubar_i = xbar_opt_i.copy(), ubar_opt_i.copy()


            self.cost_iter[ht_iter, task_id, i+1] = self._eval_cost_functions(cost_fcn, xbar_i, ubar_i, cost_fcn_params)
            self.step_size[ht_iter, task_id, i] = linesearch_step_opt
            self.solver_status[ht_iter, task_id, i] = results['status_val']
            # t1 = time.perf_counter()
            # print("Line Search time:{}".format(t1 - t0))

        return xbar_i, ubar_i, results['status']

    def scaleConstraints(self, G, h):
        largeval_idx = np.where(np.logical_or(h > 1e-2, h < -1e-2))
        # if np.any(harray[largeval_idx] < 0.):
        #     print(1)
        G[largeval_idx] = G[largeval_idx] / np.abs(h[largeval_idx]).reshape((len(largeval_idx[0]), 1))
        h[largeval_idx] = 1. * (h[largeval_idx] > 0) - 1. * (h[largeval_idx] < 0)

        smallval_idx = np.where(np.logical_and(h < 1e-2, h > -1e-2))
        h[smallval_idx] = 0.

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

    def lineSearch(self, xo, ubar, dubar, cost_fcn, cost_fcn_params, csts, csts_params):
        alphas = np.linspace(0, 1, 10)
        a_opt = None
        dubar_rp = dubar.reshape((self.N, self.nu))

        for a in alphas:
            ubar_new = ubar + a * dubar_rp
            xbar_new = self._predictTrajectories(xo, ubar_new)

            J_new = 0
            for fid, f in enumerate(cost_fcn):
                J_new += f.evaluate(xbar_new, ubar_new, cost_fcn_params[fid])

            if a == 0:
                a_opt = a
                J_opt = J_new
            else:
                # check if constraints are satisfied
                cst_feas = True
                for cst_id, cst in enumerate(csts["ineq"]):
                    feas = cst.check(xbar_new, ubar_new, *csts_params["ineq"][cst_id])
                    cst_feas = cst_feas and feas
                    if not cst_feas:
                        break

                if cst_feas:
                    if J_new - J_opt < 1e-2:
                        J_opt = J_new
                        a_opt = a

        return a_opt, J_opt

    def _eval_cost_functions(self, cost_fcn, xbar, ubar, cost_fcn_params):
        J = 0
        for id, f in enumerate(cost_fcn):
            Ji = f.evaluate(xbar, ubar, cost_fcn_params[id])
            J += Ji
        return J


class HTMPCLex(HTMPC):
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
        self.stmpc_run_time = np.zeros((self.params["HT_MaxIntvl"], task_num))

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
                        csts_params["ineq"].append([r_bars[prev_task_id][0].T, e_bars[prev_task_id]])

                t0 = time.process_time()
                xbar_lopt, ubar_lopt, status = self.solveSTMPC(xo, xbar_l.copy(), ubar_l.copy(), cost_fcn, r_bars[task_id],
                                                               csts, csts_params, ht_iter=i, task_id=task_id)
                t1 = time.process_time()
                self.stmpc_run_time[i, task_id] = t1 - t0

                e_bars_l = cost_fcn[0].e_bar_fcn(xbar_lopt.T, ubar_lopt.T, r_bars[task_id][0].T)
                e_bars.append(e_bars_l.toarray().flatten())
                xbar_l = xbar_lopt.copy()
                ubar_l = ubar_lopt.copy()

        for task_id, cost_fcn in enumerate(cost_fcns):
            self.cost_final[task_id] = self._eval_cost_functions(cost_fcn, xbar_l, ubar_l, r_bars[task_id])

        return xbar_l, ubar_l

    def solveSTMPC(self, xo, xbar, ubar, cost_fcn, cost_fcn_params, csts, csts_params, ht_iter=0, task_id=0):
        self.cost_iter[ht_iter,task_id,0] = self._eval_cost_functions(cost_fcn, xbar, ubar, cost_fcn_params)
        # f (x)
        f_eqn = cs.DM.zeros()
        for cost_id, cost in enumerate(cost_fcn):
            J_eqn = cost.J_fcn(self.x_bar_sym, self.u_bar_sym, cost_fcn_params[cost_id].T)

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
        opts = {"ipopt": {'linear_solver': "ma86", 'print_level': 0}}#, 'constr_viol_tol': 1e-5, 'tol': 1e-5}}
        S = cs.nlpsol('S', 'ipopt', nlp, opts)

        # Solve NLP
        x_init = np.hstack((xbar.flatten(), ubar.flatten()))
        res = S(x0=x_init, lbx=self.xuCst.lb, ubx=self.xuCst.ub, lbg=glb, ubg=gub)
        z_opt = res['x']

        xbar_new = z_opt[:self.nx * (self.N + 1)].reshape((self.nx, self.N + 1))
        ubar_new = z_opt[self.nx * (self.N + 1):].reshape((self.nu, self.N))

        xbar_new = xbar_new.T.toarray()
        ubar_new = ubar_new.T.toarray()

        self.cost_iter[ht_iter, task_id, 1] = self._eval_cost_functions(cost_fcn, xbar_new, ubar_new, cost_fcn_params)

        return xbar_new, ubar_new, 0

