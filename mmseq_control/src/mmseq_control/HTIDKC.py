#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:16:00 2023

@author: tracy
"""

from mmseq_control.robot import MobileManipulator3D as MM
from mmseq_control.robot import CasadiModelInterface as ModelInterface
from mmseq_control.IDKCTasks import EEPositionTracking, BasePositionTracking, JointVelocityBound, JointAngleBound, JointAccelerationBound, CollisionAvoidance

import numpy as np
import casadi as cs

class IDKC():

    def __init__(self, config):
        self.robot = MM(config)
        self.params = config
        self.QPsize = self.robot.DoF

        self.ee_pos_tracking = EEPositionTracking(self.robot, config)
        self.base_pos_tracking = BasePositionTracking(self.robot, config)
        self.qdot_bound = JointVelocityBound(self.robot, config)

    def control(self, t, robot_states, planners):
        q, _ = robot_states
        planner = planners[0]

        rd, vd = planner.getTrackingPoint(t, robot_states)

        if planner.type == "EE" and planner.ref_data_type == "Vec3":
                J, ed = self.ee_pos_tracking.linearize(q, rd, vd)
        elif planner.type == "base" and planner.ref_data_type == "Vec2":
                J, ed = self.base_pos_tracking.linearize(q, rd, vd)
        else:
            print("Planner of Type {} and Data type {} Not supported".format(planner.type, planner.ref_data_type))
            J = np.eye(self.robot.DoF)
            ed = np.zeros(self.robot.DoF)

        if self.params["solver"] == "pinv":
            qdot_d = (np.linalg.pinv(J) @ ed).toarray().flatten()
        elif self.params["solver"] == "QP":
            H = J.T @ J + self.params["ρ"] * cs.DM.eye(self.QPsize)
            g = - J.T @ ed
            qp = {}
            qp['h'] = H.sparsity()
            opts = {"error_on_fail": True,
                    "gurobi": {"OutputFlag": 0, "LogToConsole": 0, "Presolve": 1, "BarConvTol": 1e-8,
                               "OptimalityTol": 1e-6}}
            S = cs.conic('S', 'gurobi', qp, opts)

            results = S(h=H, g=g, lbx=-self.qdot_bound.ub[self.QPsize:], ubx=self.qdot_bound.ub[:self.QPsize])
            qdot_d = np.array(results['x']).squeeze()
        else:
            qdot_d = np.zeros(self.robot.DoF)
        return qdot_d, np.zeros(self.robot.DoF)

class HTIDKC():
    """ Hierarchical Task Inversed Difference Control

        Reference:
        Escande, Adrien, Nicolas Mansard, and Pierre-Brice Wieber. “Hierarchical Quadratic Programming: Fast Online
        Humanoid-Robot Motion Generation.” The International Journal of Robotics Research 33, no. 7 (June 1, 2014): 1006–28.
        https://doi.org/10.1177/0278364914521306.

    """

    def __init__(self, config):
        self.model_interface = ModelInterface(config)
        self.robot = self.model_interface.robot
        self.params = config
        self.QPsize = self.robot.DoF
        self.qdot_prev = np.zeros(self.robot.DoF)
        self.ctrl_rate = config["ctrl_rate"]

        self.ee_pos_tracking = EEPositionTracking(self.robot, config)
        self.base_pos_tracking = BasePositionTracking(self.robot, config)
        self.qdot_bound = JointVelocityBound(self.robot, config)
        self.q_bound = JointAngleBound(self.robot, config)
        self.qddot_bound = JointAccelerationBound(self.robot, config)
        self.collision_avoidance = CollisionAvoidance(self.model_interface, config)

    def control(self, t, robot_states, planners):
        q, _ = robot_states
        Js = []
        eds = []
        task_types = []
        task_names = []

        Js_ineq = []
        eds_ineq = []
        names_ineq = []

        J_joint = cs.DM.zeros((0, self.QPsize))
        ed_joint = cs.DM.zeros(0)
        joint_tasks = [self.qdot_bound, self.q_bound]
        for task in joint_tasks:
            Jq, edq = task.linearize(q, [])
            J_joint = cs.vertcat(J_joint, Jq)
            ed_joint = cs.vertcat(ed_joint, edq)

        if self.params["joint_acc_bound_enabled"]:
            Jqddot, edqddot = self.qddot_bound.linearize(q, self.qdot_prev)
            J_joint = cs.vertcat(J_joint, Jqddot)
            ed_joint = cs.vertcat(ed_joint, edqddot)

        Js_ineq.append(J_joint)
        eds_ineq.append(ed_joint)
        names_ineq.append("Joints")

        if self.params["self_collision_avoidance_enabled"] or self.params["static_obstacles_collision_avoidance_enabled"]:
            Jcol, edcol = self.collision_avoidance.linearize(q, [])
            Js_ineq.append(Jcol)
            eds_ineq.append(edcol)
            names_ineq.append("Collision")

        if self.params["one_inequality_constraint_enabled"]:
            Js.append(cs.vertcat(*Js_ineq))
            eds.append(cs.vertcat(*eds_ineq))
            task_types.append("Ineq")
            task_names.append("Joints&Collision")
        else:
            for J, ed, name in zip(Js_ineq, eds_ineq, names_ineq):
                Js.append(J)
                eds.append(ed)
                task_types.append("Ineq")
                task_names.append(name)

        for pid, planner in enumerate(planners):
            rd, vd = planner.getTrackingPoint(t, robot_states)

            if planner.type == "EE" and planner.ref_data_type == "Vec3":
                    J, ed = self.ee_pos_tracking.linearize(q, rd, vd)
                    name = self.ee_pos_tracking.name
            elif planner.type == "base" and planner.ref_data_type == "Vec2":
                    J, ed = self.base_pos_tracking.linearize(q, rd, vd)
                    name = self.base_pos_tracking.name
            else:
                print("Planner of Type {} and Data type {} Not supported".format(planner.type, planner.ref_data_type))
                J = np.eye(self.robot.DoF)
                ed = np.zeros(self.robot.DoF)
                name = "Empty"

            Js.append(J)
            eds.append(ed)
            task_types.append("Eq")
            task_names.append(name)

        qdot, ws = self.hqp(Js, eds, task_types)
        qddot = (qdot - self.qdot_prev) * self.ctrl_rate
        # bookkeeping
        self.qdot_prev = qdot
        self.ws = ws.copy()
        self.task_names = task_names.copy()

        return qdot, qddot

    def hqp(self, Js, eds, task_types):
        """ Cascaded QP for solving lexicographic quadratic programming problem
            lex-quad formulation see Eq.(22) in (Escande, 2014)
            task formulation see Eq.(69) - Eq.(71) in (Escande, 2014)

        :param Js: list of task jacobians
        :param eds: list of error feedback signal (desired twist)
        :param task_types: a list, indicating task type, equality or inequality
        :return: qdot_opt: optimal solution
        """
        Abar = cs.DM.zeros(0, self.QPsize)
        bbar = cs.DM.zeros(0)
        # if task_types[0] == "Ineq":
        #     Cbar = Js[0]
        #     dbar = eds[0]
        # else:
        Cbar = cs.DM.zeros(0, self.QPsize)
        dbar = cs.DM.zeros(0)

        w_opts = []

        for tid in range(len(task_types)):
            # if tid == 0 and task_types[0] == "Ineq":
            #     continue
            J = Js[tid]
            ed = eds[tid]

            opti = cs.Opti('conic')
            qdot = opti.variable(self.QPsize)
            w = opti.variable(J.shape[0])

            opti.minimize(0.5 * w.T @ w + 0.5 * self.params["ρ"] * qdot.T @ qdot)
            # opti.subject_to(opti.bounded(-self.qdot_bound.ub[self.QPsize:], qdot, self.qdot_bound.ub[:self.QPsize]))
            if Abar.shape[0] > 0:
                opti.subject_to(Abar @ qdot == bbar)
            if Cbar.shape[0] > 0:
                opti.subject_to(Cbar @ qdot <= dbar)
            if task_types[tid] == "Ineq":
                opti.subject_to(J @ qdot <= ed + w)
            else:
                opti.subject_to(J @ qdot == ed + w)

            p_opts = {"error_on_fail": True, "expand": True}
            s_opts = {"OutputFlag": 0, "LogToConsole": 0, "Presolve": 1, "BarConvTol": 1e-8,
                               "OptimalityTol": 1e-6}
            opti.solver('gurobi', p_opts, s_opts)

            try:
                sol = opti.solve()
            except RuntimeError:
                print(tid)
                print(dbar)
                print(dbar.toarray().flatten().reshape((2, 9)))

            w_opt = sol.value(w)
            qdot_opt = sol.value(qdot)

            w_opts.append(w_opt)
            if task_types[tid] == "Ineq":
                Cbar = cs.vertcat(Cbar, J)
                dbar = cs.vertcat(dbar, ed + w_opt)
            else:
                Abar = cs.vertcat(Abar, J)
                bbar = cs.vertcat(bbar, ed + w_opt)

        return qdot_opt, w_opts








