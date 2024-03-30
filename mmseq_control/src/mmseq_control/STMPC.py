from abc import ABC, abstractmethod
import time
import logging

import numpy as np
import casadi as cs
from mmseq_control.MPCConstraints import MotionConstraint
from mmseq_control.MPCCostFunctions import  BasePose3CostFunction, ArmJointCostFunction
from mmseq_control.robot import MobileManipulator3D as MM
from mmseq_control.robot import CasadiModelInterface as ModelInterface
from mmseq_control.HTMPC import MPC
from mmseq_utils.math import wrap_pi_array
import mobile_manipulation_central as mm

EE_POS_HOME = [1.48985, 0.17431, 0.705131]
class STMPCSQP(MPC):

    def __init__(self, config):
        super().__init__(config)
        self.MotionCst = MotionConstraint(self.dt, self.N, self.robot, "DI Motion Model")
        self.BasePose3Cost = BasePose3CostFunction(self.dt, self.N, self.robot, config["cost_params"]["BasePos2"])
        self.ArmJointCost = ArmJointCostFunction(self.dt, self.N, self.robot, config["cost_params"]["ArmJoint"])
        self.home = mm.load_home_position(config["home"])

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

        # 0.2 Get ref, cost_fcn,
        r_bars = []
        cost_fcns = []
        if len(planners) == 1:

            planner = planners[0]
            r_bar = [planner.getTrackingPoint(t + k * self.dt, (self.x_bar[k, :self.DoF], self.x_bar[k, self.DoF:]))[0]
                     for k in range(self.N + 1)]
            r_bars += [[np.array(r_bar)], [-self.u_prev]]

            if planner.type == "EE" and planner.ref_data_type == "Vec3":
                cost_fcns += [self.EEPos3BaseFrameCost, self.CtrlEffCost]
            elif planner.type == "base" and planner.ref_data_type == "Vec2":
                # cost_fcns += [self.BasePos2Cost, self.CtrlEffCost, self.EEPos3BaseFrameCost]
                cost_fcns += [self.BasePos2Cost, self.CtrlEffCost, self.ArmJointCost]
                r_qa = [self.home[3:] for k in range(self.N + 1)]
                r_bars += [[np.array(r_qa)]]
            elif planner.type == "base" and planner.ref_data_type == "Vec3":
                # cost_fcns += [self.BasePose3Cost, self.CtrlEffCost, self.EEPos3BaseFrameCost]
                # # r_e,_ = self.robot.getEE(q, base_frame=True)
                # r_eb_b = [EE_POS_HOME for k in range(self.N+1)]
                # r_bars += [[np.array(r_eb_b)]]

                cost_fcns += [self.BasePose3Cost, self.CtrlEffCost, self.ArmJointCost]
                r_qa = [self.home[3:] for k in range(self.N+1)]
                r_bars += [[np.array(r_qa)]]


            else:
                self.py_logger.warning("unknown cost type, planner # %d", id)
        else:
            for planner in planners:
                r_bar = [planner.getTrackingPoint(t + k * self.dt, (self.x_bar[k, :self.DoF], self.x_bar[k, self.DoF:]))[0]
                     for k in range(self.N + 1)]
                r_bars += [[np.array(r_bar)]]

                if planner.type == "EE" and planner.ref_data_type == "Vec3":
                    cost_fcns += [self.EEPos3Cost]
                elif planner.type == "base" and planner.ref_data_type == "Vec2":
                    # cost_fcns += [self.BasePos2Cost, self.CtrlEffCost, self.EEPos3BaseFrameCost]
                    cost_fcns += [self.BasePos2Cost]
                else:
                    self.py_logger.warning("unknown cost type, planner # %d", id)

                if planner.type == "EE":
                    self.ree_bar = r_bar 
                elif planner.type == "base":
                    self.rbase_bar = r_bar

            cost_fcns += [self.CtrlEffCost]
            r_bars += [[-self.u_prev]]

        # 0.3 get constraints
        csts = {"eq": [self.MotionCst], "ineq": []}
        csts_params = {"eq": [[xo]], "ineq": []}

        # state control soft constraint function
        if self.params["penalize_du"]:
            xuSoft_param = [[self.u_prev]]
        else:
            xuSoft_param = [[]]

        st_cost_fcn = cost_fcns + [self.xuSoftCst]
        st_cost_fcn_params = r_bars + xuSoft_param

        for name, soft_cst in self.collisionSoftCsts.items():
            st_cost_fcn += [soft_cst]
            st_cost_fcn_params += [[]]


        if self.params["ee_upward_constraint_enabled"]:
            st_cost_fcn += [self.eeUpwardSoftCst]
            st_cost_fcn_params += [[]]

        print(st_cost_fcn)
        xbar_lopt, ubar_lopt, status = self.solveSTMPCCasadi(xo, self.x_bar.copy(), self.u_bar.copy(), st_cost_fcn,
                                                             st_cost_fcn_params, csts, csts_params)
        self.cost_final = self._eval_cost_functions(cost_fcns, xbar_lopt, ubar_lopt, r_bars)
        if planner.type == "EE":
            print(self.EEPos3BaseFrameCost.evaluate(xbar_lopt, ubar_lopt, r_bars[0][0]))
        self.x_bar = xbar_lopt.copy()
        self.u_bar = ubar_lopt.copy()
        self.u_prev = self.u_bar[0].copy()
        self.v_cmd = self.x_bar[0][self.robot.DoF:].copy()

        self.ee_bar, self.base_bar = self._getEEBaseTrajectories(self.x_bar)
        print(self.v_cmd)
        return self.v_cmd, self.u_prev, self.u_bar.copy()

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
        self.cost_iter = np.zeros(self.params["ST_MaxIntvl"]+1)
        self.solver_status = np.zeros(self.params["ST_MaxIntvl"])
        self.step_size = np.zeros(self.params["ST_MaxIntvl"])

        self.cost_iter[0] = self._eval_cost_functions(cost_fcn, xbar_i, ubar_i, cost_fcn_params)

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
                results = S(h=H, g=g, a=Ac, uba=uba, lba=lba)

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

            self.cost_iter[i+1] = self._eval_cost_functions(cost_fcn, xbar_i, ubar_i,
                                                              cost_fcn_params)
            self.step_size[i] = linesearch_step_opt
            self.solver_status[i] = results['status_val']
            t1 = time.perf_counter()
            self.py_logger.log(5, "Line Search time:{}".format(t1 - t0))

        return xbar_i, ubar_i, results['status']

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
                    self.py_logger.debug(
                        "Controller: line search step acceptance condition not met {}.".format(J_xp_lin - J_xp))

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