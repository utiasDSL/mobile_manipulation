#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 22:16:32 2021

@author: tracy
"""

from mmseq_control.robot import MobileManipulator3D as MM
from mmseq_utils import parsing

import numpy as np

from mmseq_control.HQP import PrioritizedLinearSystemsNew as PLSN
# from mmseq_control.HQP import PrioritizedLinearSystems as PLS
from qpsolvers import solve_qp, qpoases_solve_qp



# class IKController():
#
#     def __init__(self, robot_class, ctrl_params):
#         self.physicsClient = bc.BulletClient(connection_mode=pyb.DIRECT)
#         self.physicsClient.setAdditionalSearchPath(pybullet_data.getDataPath())
#         self.physicsClient.loadURDF("plane.urdf", [0, 0, 0])
#         self.physicsClient.setGravity(0, 0, -9.81)
#         self.physicsClient.setTimeStep(SIM_DT)
#         self.model = robot_class(p=self.physicsClient)
#         self.model.reset_joint_configuration(ROBOT_HOME)
#         self.params = ctrl_params
#
#     def control(self, robot_states, target):
#         u = [0.0, 0, 0, 0.0, 0, 0, 0, 0, 0]
#         q, _ = robot_states
#         self.model.reset_joint_configuration(q)
#         self.model.command_velocity(u)
#         Pee, _ = self.model.link_pose()
#         Pee_t, Pb_t = target
#         Jee = self.model.jacobian(q)
#
#         err = self.params["lamda_ee"] * (Pee_t - np.array(Pee))
#         control_inputs = np.matmul(linalg.pinv(Jee[:3]), err)
#
#         return control_inputs


class IKCPrioritized:

    def __init__(self, config):
        self.robot = MM(config)
        self.fk_ee = self.robot.kinSymMdls[self.robot.tool_link_name]
        self.Jee = self.robot.jacSymMdls[self.robot.tool_link_name]
        self.ub_qdot = parsing.parse_array(config["robot"]["limits"]["state"]["upper"])[self.robot.DoF:]
        self.lb_qdot = parsing.parse_array(config["robot"]["limits"]["state"]["lower"])[self.robot.DoF:]

        self.params = config
        self.prev_control = np.zeros(9)

    def control(self, t, robot_states, planners):
        q, _ = robot_states

        Pee, _ = self.fk_ee(q)
        Pee = Pee.toarray().flatten()

        ## EE
        Abar = np.zeros((0, 9))
        bbar = np.zeros((0))
        Cbar = np.vstack((np.eye(9), -np.eye(9)))
        dbar = np.hstack((self.ub_qdot, -self.lb_qdot))
        for i, planner in enumerate(planners):
            if planner.type == "EE":
                # if planner.ref_type == "pose":
                #     Tee_d, Vee_d = planner.getTrackingPoint(t)
                #     Terr = np.matmul(np.linalg.inv(Tee), Tee_d)
                #     Vee_r = SE3.log(SE3(SO3(Terr[:3, :3]), Terr[:3, 3]))
                #     Tad = SE3.adjoint(SE3(SO3(Tee[:3, :3]), Tee[:3, 3]))
                #     Tad_alt = np.vstack((np.hstack((Tad[:3, :3], Tad[3:, :3])), np.hstack((Tad[:3, 3:], Tad[3:, 3:]))))
                #     v = self.params["lamdba_ee"] * np.matmul(Tad_alt, Vee_r)
                #     J = self.model.jacobian(q)
                #
                # elif planner.ref_type == "pos":
                Pee_d, Vee_d = planner.getTrackingPoint(t)
                vee = self.params["lambda_ee"] * (Pee_d - Pee) + Vee_d
                Jee = self.Jee(q).toarray()

                (Abar, bbar, Cbar, dbar), z = PLSN.getSubsetEq(Jee, vee, Abar, bbar, Cbar, dbar, 1e-3,
                                                               init_vals=self.prev_control)
            else:
                Pb_d, Vb_d = planner.getTrackingPoint(t)
                vb = self.params["lambda_base"] * (Pb_d - q[:2]) + Vb_d
                Jb = np.hstack((np.eye(2), np.zeros((2, 7))))
                (Abar, bbar, Cbar, dbar), z = PLSN.getSubsetEq(Jb, vb, Abar, bbar, Cbar, dbar, 1e-3,
                                                               init_vals=self.prev_control)
        H = np.eye(9)
        H[:3, :3] = np.eye(3) * self.params["wb"]
        H[3:, 3:] = np.eye(6) * self.params["wee"]
        g = np.zeros(9)

        use_pls = False
        if use_pls:
            Htask = np.matmul(J.transpose(), J)
            gtask = -np.matmul(v, J)
        if len(Abar) == 0:
            Abar = None
            bbar = None

        if use_pls:
            H += Htask
            g += gtask
            control_inputs = solve_qp(H, g, Cbar[:18], dbar[:18], None, None, initvals=self.prev_control, solver='osqp',
                                      verbose=False)

        else:
            control_inputs = solve_qp(H, g, Cbar, dbar, Abar, bbar, initvals=self.prev_control, solver='osqp',
                                      verbose=False, eps_abs=1e-3)
            if control_inputs is None:
                control_inputs = solve_qp(H, g, Cbar, dbar, Abar, bbar, initvals=self.prev_control, solver='osqp',
                                          verbose=False, eps_abs=1e-2)

            # P = matrix(H)
            # q = matrix(g)
            # G = matrix(Cbar)
            # h = matrix(dbar)
            # if Abar is not None:
            #     Ap = matrix(Abar)
            #     bp = matrix(bbar)
            # else:
            #     Ap = None
            #     bp = None
            # solvers.options['show_progress'] = False
            # solvers.options['mosek'] = {mosek.iparam.log: 0}
            # initval = {'x': matrix(self.prev_control)}
            # results = qp(P, q, G, h, Ap, bp,initvals=initval, solver='mosek')
            # control_inputs = np.array(results['x']).squeeze()

        # if control_inputs is None:
        #         control_inputs = solve_qp(H, g, Cbar, dbar, Abar, bbar, initvals=self.prev_control, solver='osqp', verbose=False, eps_abs=1e-1)
        cost = np.matmul(control_inputs, np.matmul(H, control_inputs))

        self.prev_control = control_inputs
        return control_inputs, 0
