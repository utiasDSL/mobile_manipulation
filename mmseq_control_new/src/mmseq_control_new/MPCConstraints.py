from abc import ABC, abstractmethod
import logging

import numpy as np
from scipy.linalg import block_diag, expm
import casadi as cs
from casadi.tools import struct_symMX, entry
from mmseq_control.robot import MobileManipulator3D
from mmseq_control_new.MPCCostFunctions import TrajectoryTrackingCostFunction
from mmseq_utils.casadi_struct import casadi_sym_struct

class Constraint(ABC):
    def __init__(self, nx, nu, name):
        """ MPC cost functions base class

        :param dt: discretization time step
        :param nx: state dim
        :param nu: control dim
        :param N:  prediction window
        :param name: name to identify the constraint
        """


        self.nx = nx
        self.nu = nu
        self.np = np
        self.name = name

        self.x_sym = cs.MX.sym('x', nx)
        self.u_sym = cs.MX.sym('u', nu)

        self.p_dict = None
        self.p_stuct = None
        self.p_sym = None

        self.name = name
        self.slack_enabled = False
        
        super().__init__()

    @abstractmethod
    def check(self, x, u, p):
        pass

    def get_p_dict(self, sym=True):
        if self.p_dict is None:
            return None
        else:
            if sym:
                return {key +f"_{self.name}":val for (key, val) in self.p_dict.items()}
            else:
                return {key +f"_{self.name}":cs.DM.zeros(val.shape) for (key, val) in self.p_dict.items()}
    
    def get_p_dict_default(self, stage_e=False):
        p_dict = self.get_p_dict(False)
        return p_dict

class NonlinearConstraint(Constraint):
    def __init__(self, nx, nu, ng, g_fcn, p_dict, constraint_name):
        """ nonlinear inequality constraint
                            g(x, u, p) < 0

        :param nx: state dim
        :param nu: control dim
        :param ng: constraint dim
        :param N:  prediction window
        :param g_fcn: g_fcn(x_bar_sym, u_bar_sym,*params_sym) casadi.Function
        :param params_sym: parameters [casadi struct_symMX]
        :param constraint_name: name to identify the constraint
        :param tol: constraint violation tolerence
        """
        # self.params_name = [psym.name() for psym in params_sym]
        # self.params_sym = params_sym

        super().__init__(nx, nu, constraint_name)
        self.ng = ng

        self.p_dict = p_dict
        self.p_struct = casadi_sym_struct(p_dict)
        self.p_sym = self.p_struct.cat
        
        self.g_fcn = g_fcn
        if self.g_fcn is not None:
            self.g_eqn = g_fcn(self.x_sym, self.u_sym, self.p_sym)
        else:
            self.g_eqn = None
    
    def check(self, x, u, p):
        g = self.g_fcn(x, u, p)

        return g

class HierarchicalTrackingConstraint(NonlinearConstraint):
    def __init__(self, cost_fcn_obj: TrajectoryTrackingCostFunction, name="Hierarchy"):
        """ Hierarchical Tracking constraint |e(x, u, r)| < |e_bar_p|

        :param cost_fcn_obj: an instance of class ctrl.MPCCostFunctions.TrackingCostFunction
        :param name: name to identify the constraint
        """

        nx = cost_fcn_obj.nx
        nu = cost_fcn_obj.nu
        ng = cost_fcn_obj.e_eqn.shape[0]*2
        p_dict = cost_fcn_obj.get_p_dict()
        p_dict["e_p"] = cs.MX.sym("e_p_"+name, cost_fcn_obj.e_eqn.shape[0])
        super().__init__(nx, nu, ng, None, p_dict, name)
        self.cost_fcn_obj = cost_fcn_obj

        e_eqn = cost_fcn_obj.e_fcn(self.x_sym, self.u_sym, self.p_struct["r_"+cost_fcn_obj.name])  
        e_p_abs_eqn = cs.fabs(self.p_struct["e_p"])

        self.g_eqn = cs.vertcat(e_eqn, -e_eqn) - cs.vertcat(e_p_abs_eqn, e_p_abs_eqn)
        self.g_fcn = cs.Function("g_"+self.name, [self.x_sym, self.u_sym, self.p_struct.cat], [self.g_eqn])

        self.g_grad_eqn = cs.jacobian(self.g_eqn, cs.veccat(self.u_sym, self.x_sym))
        self.g_grad_fcn = cs.Function("g_grad", [self.x_sym, self.u_sym, self.p_sym], [self.g_grad_eqn])

    def get_p_dict(self, sym=True):
        if self.p_dict is None:
            return None
        else:
            p_dict = {}
            for (key, val) in self.p_dict.items():
                if key == "e_p":
                    p_dict[key+f"_{self.name}"] = val if sym else cs.DM.zeros(val.shape)
                else:
                    p_dict[key] = val if sym else cs.DM.zeros(val.shape)

            return p_dict
    
    def get_p_dict_default(self, stage_e=False):
        p_dict = self.get_p_dict(False)
        if p_dict is not None:
            p_cost = self.cost_fcn_obj.get_p_dict_default(stage_e)
            p_dict.update(p_cost)
            
        return p_dict
    
class SignedDistanceConstraint(NonlinearConstraint):
    def __init__(self, robot_mdl, signed_distance_fcn, d_safe, name="obstacle"):
        """ Signed Distance Constraint
                   - (sd(x_k, param1, param2, ...) - d_safe) < 0

        :param robot_mdl: class mmseq_control.robot.MobileManipulator3D
        :param signed_distance_fcn: signed distance model, casadi function
        :param d_safe: safe clearance, scalar, same for all body pairs
        :param name: name of this constraint
        """
        nx = robot_mdl.ssSymMdl["nx"]
        nu = robot_mdl.ssSymMdl["nu"]
        nq = robot_mdl.q_sym.size()[0]
        ng = signed_distance_fcn.size_out(0)[0]
        p_sym = signed_distance_fcn.mx_in()[1:]
        p_name = signed_distance_fcn.name_in()[1:]
        p_dict = {name:sym for (name, sym) in zip(p_name, p_sym)}

        super().__init__(nx, nu, ng, None, p_dict, name)

        self.g_eqn = -signed_distance_fcn(self.x_sym[:nq], *[self.p_struct[k] for k in self.p_dict.keys()]) + d_safe

        self.g_fcn = cs.Function("g_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [self.g_eqn])

        self.g_grad_eqn = cs.jacobian(self.g_eqn, cs.veccat(self.u_sym, self.x_sym))
        self.g_grad_fcn = cs.Function("g_grad", [self.x_sym, self.u_sym, self.p_sym], [self.g_grad_eqn])

        # self.g_jac_eqn = cs.jacobian(self.g_eqn, cs.veccat(self.u_sym, self.x_sym))
        # self.g_jac_fcn = cs.Function("g_jac", [self.x_sym, self.u_sym, self.p_sym], [self.g_jac_eqn.T])

        self.slack_enabled = True

class SignedDistanceConstraintCBF(NonlinearConstraint):
    def __init__(self, robot_mdl, signed_distance_fcn, d_safe, name="obstacle"):
        """ Signed Distance Constraint
                    h(x) = (sd(x_k, param1, param2, ...) - d_safe) > 0
            Signed Distance CBF Constraint
                    L_fh(x) + L_gh(x) + gamma * h(x) >= 0
                    x_dot = f(x) + g(x)u

        :param robot_mdl: class mmseq_control.robot.MobileManipulator3D
        :param signed_distance_fcn: signed distance model, casadi function
        :param d_safe: safe clearance, scalar, same for all body pairs
        :param name: name of this constraint
        """
        nx = robot_mdl.ssSymMdl["nx"]
        nu = robot_mdl.ssSymMdl["nu"]
        nq = robot_mdl.q_sym.size()[0]
        ng = signed_distance_fcn.size_out(0)[0]
        p_sym = signed_distance_fcn.mx_in()[1:]
        p_name = signed_distance_fcn.name_in()[1:]
        p_dict = {name:sym for (name, sym) in zip(p_name, p_sym)}
        p_dict["gamma"] = cs.MX.sym('gamma')

        super().__init__(nx, nu, ng, None, p_dict, name)

        h_eqn = signed_distance_fcn(self.x_sym[:nq], *[self.p_struct[k] for k in p_name]) - d_safe
        # h_grad_eqn = cs.jacobian(h_eqn, self.x_sym)
        # h_grad_eqn_normalized_list = [h_grad_eqn, h_grad_eqn/cs.norm_2(h_grad_eqn)]
        # h_grad_eqn_normalized = cs.conditional(cs.norm_2(h_grad_eqn)>0.01,
        #                                           h_grad_eqn_normalized_list, 0 , False)
        
        # self.g_eqn = -(h_grad_eqn @ robot_mdl.ssSymMdl["fmdl"](self.x_sym, self.u_sym) + self.p_dict["gamma"]*h_eqn)
        h_grad_eqn = cs.jacobian(h_eqn, self.x_sym)
        gammas = [self.p_dict["gamma"]*2, self.p_dict["gamma"]]
        gamma = cs.conditional(h_eqn>0, gammas, 0, False)
        self.g_eqn = -(h_grad_eqn[:, :nq] @ self.x_sym[nq:] + self.p_dict["gamma"]*h_eqn)/5

        self.g_fcn = cs.Function("g_"+self.name+"_CBF", [self.x_sym, self.u_sym, self.p_sym], [self.g_eqn])

        self.g_grad_eqn = cs.jacobian(self.g_eqn, cs.veccat(self.u_sym, self.x_sym))
        self.g_grad_fcn = cs.Function("g_grad", [self.x_sym, self.u_sym, self.p_sym], [self.g_grad_eqn])
        self.gamma_fcn = cs.Function("gamma", [self.x_sym, self.u_sym, self.p_sym], [gamma])

        # self.h_fcn = cs.Function("h_fcn", [self.x_sym, self.u_sym, self.p_sym], [h_eqn])
        # self.xdot_fcn = cs.Function("xdot_fcn", [self.x_sym, self.u_sym, self.p_sym], [robot_mdl.ssSymMdl["fmdl"](self.x_sym, self.u_sym)])
        # self.h_grad_fcn = cs.Function("h_grad_fcn", [self.x_sym, self.u_sym, self.p_sym], [h_grad_eqn])

        # h_grad_fd_fcn = cs.Function("h_grad_fd_fcn", [self.x_sym, self.u_sym, self.p_sym], [h_grad_eqn],{"enable_fd":True})
        # h_hessian_fd_eqn = cs.jacobian(h_grad_fd_fcn(self.x_sym, self.u_sym, self.p_sym).T, self.x_sym)
        # self.h_hess_fd_fcn = cs.Function("h_hess_fd_fcn", [self.x_sym, self.u_sym, self.p_sym], [h_hessian_fd_eqn])

        # g_fd_fcn = cs.Function("g_"+self.name+"_CBF", [self.x_sym, self.u_sym, self.p_sym], [self.g_eqn],{"enable_fd":True})
        # g_hess_fd_eqn,g_grad_fd_eqn = cs.hessian(g_fd_fcn(self.x_sym, self.u_sym, self.p_sym), self.x_sym)
        # self.g_hess_fd_fcn = cs.Function("g_hess_fd_fcn", [self.x_sym, self.u_sym, self.p_sym], [g_hess_fd_eqn])

        self.slack_enabled = True

    def check_gamma(self, x, u, p):
        gamma = self.gamma_fcn(x, u, p)

        return gamma
    
class StateBoxConstraints(NonlinearConstraint):
    def __init__(self, robot_mdl, name="state"):
        """ Signed Distance Constraint
                   lb_x < x < ub_x

        :param robot_mdl: class mmseq_control.robot.MobileManipulator3D
        :param signed_distance_fcn: signed distance model, casadi function
        :param d_safe: safe clearance, scalar, same for all body pairs
        :param name: name of this constraint
        """
        nx = robot_mdl.ssSymMdl["nx"]
        nu = robot_mdl.ssSymMdl["nu"]
        nq = robot_mdl.q_sym.size()[0]
        ng = nx*2
        p_dict = {}
        super().__init__(nx, nu, ng, None, p_dict, name)

        self.g_eqn = cs.vertcat(self.x_sym - robot_mdl.ssSymMdl["ub_x"],
                                robot_mdl.ssSymMdl["lb_x"] - self.x_sym)
        self.g_fcn = cs.Function("g_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [self.g_eqn])

class ControlBoxConstraints(NonlinearConstraint):
    def __init__(self, robot_mdl, name="control"):
        """ Signed Distance Constraint
                   lb_u < u < ub_u

        :param robot_mdl: class mmseq_control.robot.MobileManipulator3D
        :param signed_distance_fcn: signed distance model, casadi function
        :param d_safe: safe clearance, scalar, same for all body pairs
        :param name: name of this constraint
        """
        nx = robot_mdl.ssSymMdl["nx"]
        nu = robot_mdl.ssSymMdl["nu"]
        ng = nu*2
        p_dict = {}
        super().__init__(nx, nu, ng, None, p_dict, name)

        self.g_eqn = cs.vertcat(self.u_sym - robot_mdl.ssSymMdl["ub_u"],
                                robot_mdl.ssSymMdl["lb_u"] - self.u_sym)
        self.g_fcn = cs.Function("g_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [self.g_eqn])