from abc import ABC, abstractmethod
import casadi as cs

import rospkg
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag
from liecasadi import SO3

from mmseq_control.robot import MobileManipulator3D
from mmseq_utils.casadi_struct import casadi_sym_struct
from mmseq_utils.math import casadi_SO2, casadi_SO3_log

class RBF:
    mu_sym = cs.MX.sym('mu')
    zeta_sym = cs.MX.sym('zeta')
    h_sym = cs.MX.sym('h')

    B_eqn_list = [-mu_sym * cs.log(h_sym),
                  mu_sym * (0.5 * (((h_sym - 2 * zeta_sym) / zeta_sym) ** 2 - 1) - cs.log(zeta_sym))]
    s_eqn = h_sym < zeta_sym
    B_eqn = cs.conditional(s_eqn, B_eqn_list, 0, False)
    B_fcn = cs.Function("B_fcn", [h_sym, mu_sym, zeta_sym], [B_eqn])

    B_hess_eqn, B_grad_eqn = cs.hessian(B_eqn, h_sym)
    B_hess_fcn = cs.Function("ddBddh_fcn", [h_sym, mu_sym, zeta_sym], [B_hess_eqn])
    B_grad_fcn = cs.Function("dBdh_fcn", [h_sym, mu_sym, zeta_sym], [B_grad_eqn])


class CostFunctions(ABC):
    def __init__(self, nx: int, nu:int, name: str="MPCCost"):
        """ MPC cost functions base class
                        \mathcal{J}
        :param nx: state dim
        :param nu: control dim
        :param xsym: state, CasADi symbolic variable
        :param usym: control, CasADi symbolic variable
        :param psym: params, CasADi symbolic variable
        :param name: name, string

        """

        self.nx = nx
        self.nu = nu
        self.name = name

        self.x_sym = cs.MX.sym('x', nx)
        self.u_sym = cs.MX.sym('u', nu)

        self.p_sym = None
        self.p_dict = None
        self.p_struct = None
        self.J_eqn = None
        self.J_fcn = None
        self.H_approx_eqn = None
        self.H_approx_fcn = None

        super().__init__()
        
    def evaluate(self, x, u, p):
        if self.J_fcn is not None:
            return self.J_fcn(x,u,p).toarray().flatten()
        else:
            return None

    def get_p_dict(self):
        if self.p_dict is None:
            return None
        else:
            return {key +f"_{self.name}":val for (key, val) in self.p_dict.items()}
    
    def get_custom_H_fcn(self):
        return self.H_approx_fcn

class NonlinearLeastSquare(CostFunctions):
    def __init__(self, nx:int, nu:int, nr:int, f_fcn:cs.Function, name):
        """ Nonlinear least square cost function
            J = 1/2 ||f_fcn(x) - r||^2_W

        :param dt: discretization time step
        :param nx: state dim
        :param nu: control dim
        :param N:  prediction window
        :param nr: ref dim
        :param f_fcn: output/observation function, casadi.Function
        :param params: cost function params
        """
        super().__init__(nx, nu, name)
        self.nr = nr
        self.f_fcn = f_fcn.expand()

        self.p_dict = {"r": cs.MX.sym("r_"+self.name, self.nr),
                       "W": cs.MX.sym("W_"+self.name, self.nr, self.nr)}
        self.p_struct = casadi_sym_struct(self.p_dict)
        self.p_sym = self.p_struct.cat

        self.W = self.p_struct["W"]
        self.r = self.p_struct['r']
        self._setupSymMdl()

    def _setupSymMdl(self):
        y = self.f_fcn(self.x_sym)
        e = y - self.r
        self.J_eqn = 0.5 * e.T @ self.W @ e
        self.J_fcn = cs.Function("J_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [self.J_eqn], ["x", "u", "r"], ["J"]).expand()
        dedx = cs.jacobian(e, self.x_sym)
        self.H_approx_eqn = cs.diagcat(cs.MX.zeros(self.nu, self.nu), dedx.T @ self.W @ dedx)
        self.H_approx_fcn = cs.Function("H_approx_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [self.H_approx_eqn], ["x", "u", "r"], ["H_approx"]).expand()

class TrajectoryTrackingCostFunction(NonlinearLeastSquare):
    def __init__(self, nx: int, nu: int, nr: int, f_fcn: cs.Function, name):
        super().__init__(nx, nu, nr, f_fcn, name)
        self.e_eqn = self.f_fcn(self.x_sym) - self.r
        self.e_fcn = cs.Function("e_"+self.name, [self.x_sym, self.u_sym, self.r], [self.e_eqn], ["x", "u", "r"], ["e"]).expand()

    def get_e(self, x, u, r):
        return self.e_fcn(x, u ,r).toarray().flatten()

class EEPos3CostFunction(TrajectoryTrackingCostFunction):
    def __init__(self, robot_mdl, params):
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        nr = 3
        fk_ee = robot_mdl.kinSymMdls[robot_mdl.tool_link_name]
        f_fcn = cs.Function("fee", [robot_mdl.x_sym], [fk_ee(robot_mdl.q_sym)[0]])
        super().__init__(nx, nu, nr, f_fcn, "EEPos3")

class EEVel3CostFunction(TrajectoryTrackingCostFunction):
    def __init__(self, robot_mdl, params):
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        nr = 3
        jac_ee = robot_mdl.jacSymMdls[robot_mdl.tool_link_name]
        f_fcn = cs.Function("vee", [robot_mdl.x_sym], [jac_ee(robot_mdl.q_sym) @ robot_mdl.v_sym])
        super().__init__(nx, nu, nr, f_fcn, "EEVel3")

class EEPos3BaseFrameCostFunction(TrajectoryTrackingCostFunction):
    def __init__(self, robot_mdl, params):
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        nr = 3
        fk_ee = robot_mdl._getFk(robot_mdl.tool_link_name, base_frame=True)
        f_fcn = cs.Function("fee", [robot_mdl.x_sym], [fk_ee(robot_mdl.q_sym)[0]])
        super().__init__(nx, nu, nr, f_fcn, "EEPos3BaseFrame")

class PoseSE3CostFunction(CostFunctions):
    def __init__(self, nx, nu, f_fcn, name="PoseSE3"):
        super().__init__(nx, nu, name)

        self.nr = 6
        self.p_dict = {"r": cs.MX.sym("r_"+self.name, self.nr),
                       "W": cs.MX.sym("W_"+self.name, self.nr, self.nr)}
        self.p_struct = casadi_sym_struct(self.p_dict)
        self.p_sym = self.p_struct.cat

        self.W = self.p_struct["W"]
        self.r = self.p_struct['r']
        r_pos = self.r[:3]
        r_rot_euler = self.r[3:]

        # position: 3d vector, rot: rotational matrix
        pos, rot = f_fcn(self.x_sym)

        e_pos = pos - r_pos
        orn = SO3.from_matrix(rot)      # conversion from rotation matrix to quaternion
        rot_inv = SO3(cs.vertcat(-orn.xyzw[:3], orn.xyzw[3])).as_matrix()
        r_rot = SO3.from_euler(r_rot_euler).as_matrix()

        e_rot = casadi_SO3_log(rot_inv @ r_rot)

        self.e_eqn = cs.vertcat(e_pos, e_rot)
        self.J_eqn = 0.5 * self.e_eqn.T @ self.W @ self.e_eqn
        # sigma = 2
        # self.J_eqn = 0.5 * cs.log(1 + (self.J_eqn**2) / (sigma**2)) * 20

        self.J_fcn = cs.Function("J_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [self.J_eqn], ["x", "u", "r"], ["J"]).expand()
        
        self.e_fcn = cs.Function("e_"+self.name, [self.x_sym, self.u_sym, self.r], [self.e_eqn], ["x", "u", "r"], ["e"]).expand()
        dedx = cs.jacobian(self.e_eqn, self.x_sym)
        self.H_approx_eqn = cs.diagcat(cs.MX.zeros(self.nu, self.nu), dedx.T @ self.W @ dedx)
        self.H_approx_fcn = cs.Function("H_approx_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [self.H_approx_eqn], ["x", "u", "r"], ["H_approx"]).expand()
        
        self.orn_fcn = cs.Function("e_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [orn.xyzw], ["x", "u", "r"], ["e"]).expand()
        self.rot_inv_fcn = cs.Function("e_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [rot_inv], ["x", "u", "r"], ["e"]).expand()
        self.r_rot_fcn = cs.Function("e_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [r_rot], ["x", "u", "r"], ["e"]).expand()
        self.rot_err_fcn = cs.Function("e_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [rot_inv @ r_rot], ["x", "u", "r"], ["e"]).expand()

    def get_e(self, x, u, r):
        return self.e_fcn(x, u ,r).toarray().flatten()
    
class EEPoseSE3CostFunction(PoseSE3CostFunction):
    def __init__(self, robot_mdl, params):
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]

        fk_ee = robot_mdl._getFk(robot_mdl.tool_link_name)
        p_ee, rot_ee = fk_ee(robot_mdl.q_sym)
        f_fcn = cs.Function("fee", [robot_mdl.x_sym], [p_ee, rot_ee])
        super().__init__(nx, nu, f_fcn,"EEPose")

class EEPoseSE3BaseFrameCostFunction(PoseSE3CostFunction):
    def __init__(self, robot_mdl, params):
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]

        fk_ee = robot_mdl._getFk(robot_mdl.tool_link_name, True)
        p_ee, rot_ee = fk_ee(robot_mdl.q_sym)
        f_fcn = cs.Function("fee", [robot_mdl.x_sym], [p_ee, rot_ee])
        super().__init__(nx, nu, f_fcn,"EEPoseBaseFrame")

class ArmJointCostFunction(TrajectoryTrackingCostFunction):
    def __init__(self, robot_mdl, params):

        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        nr = 6
        f_fcn = cs.Function("f_qa", [robot_mdl.x_sym], [robot_mdl.qa_sym])

        super().__init__(nx, nu, nr, f_fcn, "ArmJoint")


class BasePos2CostFunction(TrajectoryTrackingCostFunction):
    def __init__(self, robot_mdl, params):
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        nr = 2
        fk_b = robot_mdl.kinSymMdls["base"]
        f_fcn = cs.Function("fb", [robot_mdl.x_sym], [fk_b(robot_mdl.q_sym)[0]])

        super().__init__(nx, nu, nr, f_fcn, "BasePos2")

class BaseVel2CostFunction(TrajectoryTrackingCostFunction):
    def __init__(self, robot_mdl, params):
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        nr = 2
        jac_b = robot_mdl.jacSymMdls["base"]
        
        f_fcn = cs.Function("fb", [robot_mdl.x_sym], [jac_b(robot_mdl.q_sym) @ robot_mdl.v_sym])

        super().__init__(nx, nu, nr, f_fcn, "BaseVel2")

class BasePos3CostFunction(TrajectoryTrackingCostFunction):
    def __init__(self, robot_mdl, params):
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        nr = 3
        fk_b = robot_mdl.kinSymMdls["base"]
        xy, h = fk_b(robot_mdl.q_sym)
        f_fcn = cs.Function("fb", [robot_mdl.x_sym], [cs.vertcat(xy, h)])

        super().__init__(nx, nu, nr, f_fcn, "BasePos3")

class BasePoseSE2CostFunction(CostFunctions):
    def __init__(self, robot_mdl, params):
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        super().__init__(nx, nu,"BasePoseSE2")

        self.nr = 3
        self.p_dict = {"r": cs.MX.sym("r_"+self.name, self.nr),
                       "W": cs.MX.sym("W_"+self.name, self.nr, self.nr)}
        self.p_struct = casadi_sym_struct(self.p_dict)
        self.p_sym = self.p_struct.cat

        self.W = self.p_struct["W"]
        self.r = self.p_struct['r']

        fk_b = robot_mdl.kinSymMdls["base"]
        # Bug warning
        xy, h = self.x_sym[:2], self.x_sym[2]

        # position
        e_pos = xy - self.r[:2]
        
        # heading
        Rinv = casadi_SO2(-h)
        Rd = casadi_SO2(self.r[2])
        Rerr = Rinv @ Rd
        
        e_h = cs.atan2(Rerr[1,0], Rerr[0, 0])

        self.e_eqn = cs.vertcat(e_pos, e_h)
        self.J_eqn = 0.5 * self.e_eqn.T @ self.W @ self.e_eqn

        self.J_fcn = cs.Function("J_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [self.J_eqn], ["x", "u", "r"], ["J"]).expand()
        self.e_fcn = cs.Function("e_"+self.name, [self.x_sym, self.u_sym, self.r], [self.e_eqn], ["x", "u", "r"], ["J"]).expand()
        
        dedx = cs.jacobian(self.e_eqn, self.x_sym)
        self.H_approx_eqn = cs.diagcat(cs.MX.zeros(self.nu, self.nu), dedx.T @ self.W @ dedx)
        self.H_approx_fcn = cs.Function("H_approx_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [self.H_approx_eqn], ["x", "u", "r"], ["H_approx"]).expand()

    def get_e(self, x, u, r):
        return self.e_fcn(x, u ,r).toarray().flatten()
    
class BaseVel3CostFunction(TrajectoryTrackingCostFunction):
    def __init__(self, robot_mdl, params):
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        nr = 3
        
        f_fcn = cs.Function("fb", [robot_mdl.x_sym], [robot_mdl.vb_sym])

        super().__init__(nx, nu, nr, f_fcn, "BaseVel3")

class ControlEffortCostFunction(CostFunctions):
    def __init__(self, robot_mdl, params):

        ss_mdl = robot_mdl.ssSymMdl
        self.params = params
        super().__init__(ss_mdl["nx"], ss_mdl["nu"], "ControlEffort")

        self.p_dict = {"Qqb": cs.MX.sym("Qqb_"+self.name, 3),
                       "Qqa": cs.MX.sym("Qqa_"+self.name, 6),
                       "Qvb": cs.MX.sym("Qvb_"+self.name, 3),
                       "Qva": cs.MX.sym("Qva_"+self.name, 6),
                       "Qub": cs.MX.sym("Qub_"+self.name, 3),
                       "Qua": cs.MX.sym("Qua_"+self.name, 6),}
        self.p_struct = casadi_sym_struct(self.p_dict)
        self.p_sym = self.p_struct.cat

        self._setupSymMdl()


    def _setupSymMdl(self):
        Qq = cs.vertcat(self.p_struct["Qqb"], self.p_struct["Qqa"], self.p_struct["Qvb"], self.p_struct["Qva"])
        Qx = cs.diag(Qq)

        Qu = cs.vertcat(self.p_struct["Qub"], self.p_struct["Qua"])
        Qu = cs.diag(Qu)

        self.J_eqn = 0.5 * self.x_sym.T @ Qx @ self.x_sym + 0.5 * self.u_sym.T @ Qu @ self.u_sym
        self.J_fcn = cs.Function("J_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [self.J_eqn], ["x", "u", "r"], ["J"]).expand()
        self.H_approx_eqn = cs.diagcat(Qu, Qx)
        self.H_approx_fcn = cs.Function("H_approx_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [self.H_approx_eqn], ["x", "u", "r"], ["H_approx"]).expand()

class SoftConstraintsRBFCostFunction(CostFunctions):
    def __init__(self, mu, zeta, cst_obj, name="SoftConstraint", expand=True):
        super().__init__(cst_obj.nx, cst_obj.nu, name)

        self.mu = mu
        self.zeta = zeta

        self.p_dict = cst_obj.get_p_dict()
        self.p_struct = casadi_sym_struct(self.p_dict)
        self.p_sym = self.p_struct.cat

        self.h_eqn = -cst_obj.g_fcn(self.x_sym, self.u_sym, self.p_sym)
        self.dhdz_eqn = -cst_obj.g_grad_fcn(self.x_sym, self.u_sym, self.p_sym)
        self.nh = self.h_eqn.shape[0]

        J_eqn_list = [RBF.B_fcn(self.h_eqn[k], self.mu, self.zeta) for k in range(self.nh)]
        self.J_eqn = sum(J_eqn_list)
        self.J_vec_eqn = cs.vertcat(*J_eqn_list)
        ddBddh_eqn_list = [RBF.B_hess_fcn(self.h_eqn[k], self.mu, self.zeta) for k in range(self.nh)]
        ddBddh_eqn = cs.diag(cs.vertcat(*ddBddh_eqn_list))
        self.H_approx_eqn = self.dhdz_eqn.T @ ddBddh_eqn @ self.dhdz_eqn

        self.h_fcn = cs.Function("h_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [self.h_eqn])
        self.J_fcn = cs.Function("J_" + self.name, [self.x_sym, self.u_sym, self.p_sym], [self.J_eqn])
        self.J_vec_fcn = cs.Function("J_vec_" + self.name, [self.x_sym, self.u_sym, self.p_sym], [self.J_vec_eqn])
        self.H_approx_fcn = cs.Function("H_approx_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [self.H_approx_eqn], ["x", "u", "r"], ["H_approx"])
            
        if expand:
            self.h_fcn = self.h_fcn.expand()
            self.J_fcn = self.J_fcn.expand()
            self.J_vec_fcn = self.J_vec_fcn.expand()
            self.H_approx_fcn = self.H_approx_fcn.expand()
        
    def evaluate_vec(self, x, u, p):
        return self.J_vec_fcn(x, u, p).toarray().flatten()
    
    def get_p_dict(self):
        return self.p_dict

class RegularizationCostFunction(CostFunctions):
    def __init__(self, nx: int, nu: int, name="Regularization"):
        super().__init__(nx, nu, name)
        self.p_dict = {"eps": cs.MX.sym("eps_reg", 1)}
        self.p_struct = casadi_sym_struct(self.p_dict)
        self.p_sym = self.p_struct.cat
        self.J_eqn = 0.5 * (self.x_sym.T @ self.x_sym + self.u_sym.T @ self.u_sym) * self.p_struct["eps"]
        self.J_fcn = cs.Function('J_' + self.name, [self.x_sym, self.u_sym, self.p_sym], [self.J_eqn])
        self.H_approx_eqn = cs.MX.eye(self.nx+self.nu) * self.p_struct["eps"]
        self.H_approx_fcn = cs.Function("H_approx_"+self.name, [self.x_sym, self.u_sym, self.p_sym], [self.H_approx_eqn], ["x", "u", "eps"], ["H_approx"])

class ManipulabilityCostFunction(CostFunctions):
    def __init__(self, robot_mdl: MobileManipulator3D, name: str = "Manipulability"):
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        super().__init__(nx, nu, name)

        self.p_dict = {"w": cs.Mx.sym('w', 1)}
        self.p_struct = casadi_sym_struct(self.p_dict)
        self.p_sym = self.p_struct.cat

        manipulability_fcn = robot_mdl.manipulability_fcn
        self.J_eqn = robot_mdl.arm_manipulability_fcn(robot_mdl.q_sym) **2 * self.p_dict["w"] * 0.5
        self.J_fcn = cs.Function("fee", [robot_mdl.x_sym], [self.J_eqn])
