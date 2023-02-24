from abc import ABC, abstractmethod
import logging

import numpy as np
from scipy.linalg import block_diag, expm
import casadi as cs
from mmseq_control.robot import MobileManipulator3D
from mmseq_control.MPCCostFunctions import BasePos2CostFunction, EEPos3CostFunction

class Constraint(ABC):
    def __init__(self, dt, nx, nu, N, name):
        """ MPC cost functions base class

        :param dt: discretization time step
        :param nx: state dim
        :param nu: control dim
        :param N:  prediction window
        :param name: name to identify the constraint
        """


        self.dt = dt
        self.nx = nx
        self.nu = nu
        self.N = N
        self.name = name

        self.QPsize = self.nx * (self.N + 1) + self.nu * self.N
        self.x_bar_sym = cs.MX.sym('x_bar', self.nx, self.N + 1)
        self.u_bar_sym = cs.MX.sym('u_bar', self.nu, self.N)
        self.z_bar_sym = cs.vertcat(cs.vec(self.x_bar_sym), cs.vec(self.u_bar_sym))

        self.py_logger = logging.getLogger("MPC")
        super().__init__()

    @abstractmethod
    def linearize(self, x_bar, u_bar, *params):
        pass

    @abstractmethod
    def check(self, x_bar, u_bar, *params):
        pass

class MotionConstraint(Constraint):
    def __init__(self, dt, N, robot_mdl, name="Motion Model"):
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        super().__init__(dt, nx, nu, N, name)

        self.nh = (N+1) * nx
        self.xo_sym = cs.MX.sym("xo", nx)
        self.fmdlc_fcn = ss_mdl["fmdl"]
        self.fmdld_fcn = self._discretizefmdl(ss_mdl)
        self.h_eqn, self.h_fcn = self._setUpConstraintSymMdl()
        self.grad_eqn = cs.jacobian(self.h_eqn, self.z_bar_sym)
        self.grad_fcn = cs.Function("dhdz", [self.z_bar_sym, self.xo_sym], [self.grad_eqn])

        self.params_name = ["xo"]
        self.tol = 1e-5

    def _setUpConstraintSymMdl(self):
        fmdl_pred_multishooting_fcn = self.fmdld_fcn.map(self.N)
        x_next_bar_eqn = fmdl_pred_multishooting_fcn(self.x_bar_sym[:, :-1], self.u_bar_sym)
        hmdl_eqn = cs.vec(self.x_bar_sym - cs.horzcat(self.xo_sym, x_next_bar_eqn))
        hmdl_fcn = cs.Function("hmdl", [self.x_bar_sym, self.u_bar_sym, self.xo_sym], [hmdl_eqn], ["x_bar", "u_bar", "xo"], ["hmdl"])

        return hmdl_eqn, hmdl_fcn


    def _discretizefmdl(self, ss_mdl):
        fcts_fcn = ss_mdl["fmdl"]
        x_sym = ss_mdl["x"]
        u_sym = ss_mdl["u"]

        if "linear" in ss_mdl["mdl_type"]:
            fdsc_fcn = ss_mdl["fmdlk"]
            return fdsc_fcn

    def linearize(self, x_bar, u_bar, *params):
        z_bar = np.hstack((x_bar.flatten(), u_bar.flatten()))
        A = self.grad_fcn(z_bar, *params)
        b = self.h_fcn(x_bar.T, u_bar.T, *params)

        return A, b

    def check(self, x_bar, u_bar, *params):
        h = self.h_fcn(x_bar.T, u_bar.T, *params)

        indx = np.where(h > self.tol)[0]
        if len(indx) > 0:
            self.py_logger.debug(self.name + " Constraint is violated at {}".format(indx))
            return False
        else:
            return True

# TODO rename to nonlinear inequality constraint
class NonlinearConstraint(Constraint):
    def __init__(self, dt, nx, nu, ng, N, g_fcn, params_sym, constraint_name, tol=1e-5):
        """

        :param dt: discretization time step
        :param nx: state dim
        :param nu: control dim
        :param ng: constraint dim
        :param N:  prediction window
        :param g_fcn: g_fcn(x_bar_sym, u_bar_sym,*params_sym) casadi.Function
        :param params_sym: list of parameters [casadi sym]
        :param constraint_name: name to identify the constraint
        :param tol: constraint violation tolerence
        """
        super().__init__(dt, nx, nu, N, constraint_name)
        self.ng = ng
        self.g_fcn = g_fcn
        self.params_name = [psym.name() for psym in params_sym]
        self.params_sym = params_sym
        self.tol = tol
        self.g_eqn = self.g_fcn(self.x_bar_sym, self.u_bar_sym, *self.params_sym)
        self.grad_eqn = cs.jacobian(self.g_eqn, self.z_bar_sym)
        self.grad_fcn = cs.Function('dgdz_'+constraint_name, [self.z_bar_sym] + self.params_sym, [self.grad_eqn], ["z_bar"]+self.params_name, ["dgdz"])

    def linearize(self, x_bar, u_bar, *params):
        z_bar = np.hstack((x_bar.flatten(), u_bar.flatten()))
        C = self.grad_fcn(z_bar, *params)
        d = self.g_fcn(x_bar.T, u_bar.T, *params)

        return C, d

    def check(self, x_bar, u_bar, *params):
        g = self.g_fcn(x_bar.T, u_bar.T, *params)

        indx = np.where(g > self.tol)[0]
        if len(indx) > 0:
            self.py_logger.debug(self.name + " Constraint is violated at {}".format(indx))
            return False
        else:
            return True

class HierarchicalTrackingConstraint(NonlinearConstraint):
    def __init__(self, cost_fcn_obj, name="Hierarchy"):
        """ Hierarchy constraint |e_bar(x_bar, u_bar, r_bar)| < |e_bar_p|

        :param cost_fcn_obj: an instance of class ctrl.MPCCostFunctions.TrackingCostFunction
        :param name: name to identify the constraint
        """
        dt = cost_fcn_obj.dt
        N = cost_fcn_obj.N
        nx = cost_fcn_obj.nx
        nu = cost_fcn_obj.nu
        ng = cost_fcn_obj.e_bar_eqn.size(1)

        e_bar_eqn = cost_fcn_obj.e_bar_eqn
        e_bar_p_sym = cs.MX.sym("e_bar_p", ng)
        e_bar_p_abs_eqn = cs.fabs(e_bar_p_sym)
        g_eqn = cs.vertcat(e_bar_eqn, -e_bar_eqn) - cs.vertcat(e_bar_p_abs_eqn, e_bar_p_abs_eqn)
        g_fcn = cs.Function("g_"+name, [cost_fcn_obj.x_bar_sym, cost_fcn_obj.u_bar_sym, cost_fcn_obj.r_bar_sym, e_bar_p_sym], [g_eqn])
        params_sym = [cost_fcn_obj.r_bar_sym, e_bar_p_sym]

        super().__init__(dt, nx, nu, ng, N, g_fcn, params_sym, name+"_h", 1e-2)

class StateControlBoxConstraint(NonlinearConstraint):
    def __init__(self, dt, robot_mdl, N):
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        name = "xubox"
        Constraint.__init__(self, dt, nx, nu, N, name)

        ng = self.z_bar_sym.size(1)
        self.ub = np.array(ss_mdl["ub_x"] * (N+1) + ss_mdl["ub_u"] * N)
        self.lb = np.array(ss_mdl["lb_x"] * (N+1) + ss_mdl["lb_u"] * N)
        g_eqn = cs.vertcat(self.z_bar_sym, -self.z_bar_sym) - np.expand_dims(np.hstack((self.ub, -self.lb)),-1)
        g_fcn = cs.Function('g_'+name, [self.x_bar_sym, self.u_bar_sym], [g_eqn], ["x_bar", "u_bar"], ['g_'+name])

        super().__init__(dt, nx, nu, ng, N, g_fcn, [], name, tol=1e-2)

def testBoxConstraint():
    print("Testing Box Constraint")
    dt = 0.1
    N = 1
    # robot mdl
    robot = MobileManipulator3D()
    nx = robot.ssSymMdl["nx"]
    nu = robot.ssSymMdl["nu"]
    Qpsize = (N+1)*nx + N * nu
    cst = StateControlBoxConstraint(dt, robot, N)
    x_bar = np.ones((N + 1, nx))
    u_bar = np.ones((N, nu))
    z_bar = np.hstack((x_bar.flatten(), u_bar.flatten()))

    lb = np.array(robot.ssSymMdl["lb_x"] * (N+1) + robot.ssSymMdl["lb_u"] * N)
    ub = np.array(robot.ssSymMdl["ub_x"] * (N+1) + robot.ssSymMdl["ub_u"] * N)
    Csym, dsym = cst.linearize(x_bar, u_bar)

    Cnum = np.vstack((np.eye(Qpsize), -np.eye(Qpsize)))
    dnum = np.hstack((-ub, lb)) + Cnum @ z_bar

    Cdiff = np.linalg.norm(Cnum - Csym)
    ddiff = np.linalg.norm(dnum - dsym)
    print("C diff:{}".format(Cdiff))
    print("d diff:{}".format(ddiff))

def testHiearchicalConstraint():
    dt = 0.1
    N = 1
    # robot mdl
    robot = MobileManipulator3D()

    # cost function params
    cost_params = {}
    cost_params["EE"] = {"Qk": 1., "P": 1.}
    cost_params["base"] = {"Qk": 1., "P": 1.}
    cost_params["effort"] = {"Qqa": 0., "Qqb": 0.,
                             "Qva": 1e-2, "Qvb": 2e-2,
                             "Qua": 1e-1, "Qub": 1e-1}

    cost_base = BasePos2CostFunction(dt, N, robot, cost_params["base"])
    cost_ee = EEPos3CostFunction(dt, N, robot, cost_params["EE"])
    base_h_cst = HierarchicalTrackingConstraint(cost_base, "base")
    ee_h_cst = HierarchicalTrackingConstraint(cost_ee, "ee")


def testNonlinearConstraint():
    dt = 0.1
    nx = 4
    nu = 2
    N = 1
    ng = 2
    QPsize = nx * (N+1) + nu * N

    x_bar_sym = cs.MX.sym("x_bar", nx, (N+1))
    u_bar_sym = cs.MX.sym("u_bar", nu, N)
    p_sym = [cs.MX.sym("q"), cs.MX.sym("r")]
    x_idx =[np.random.randint(nx), np.random.randint(N+1)]
    u_idx = [np.random.randint(nu), np.random.randint(N)]
    g_eqn = cs.vertcat(x_bar_sym[x_idx[0], x_idx[1]]*p_sym[0], u_bar_sym[u_idx[0],u_idx[1]]*p_sym[1])
    g_fcn = cs.Function("g_test", [x_bar_sym, u_bar_sym] + p_sym, [g_eqn])

    test_cst = NonlinearConstraint(dt, nx, nu, ng, N, g_fcn, p_sym, "test")
    x_bar = np.ones((N+1, nx))
    u_bar = np.ones((N, nu))
    p = [1, 4]
    C_sym,d_sym = test_cst.linearize(x_bar, u_bar, *p)

    C_num = np.zeros((ng, QPsize))
    x_idx_flat = x_idx[1] * nx + x_idx[0]
    u_idx_flat = (N+1) * nx + u_idx[1] * nu + u_idx[0]

    C_num[0, x_idx_flat] = p[0]
    C_num[1, u_idx_flat] = p[1]
    d_num = np.array(p)

    C_diff = np.linalg.norm(C_num - C_sym)
    d_diff = np.linalg.norm(d_num - d_sym)
    print("Testing Nonlinear Constraint")
    print("C diff:{}".format(C_diff))
    print("d diff:{}".format(d_diff))
    print("x_id:{}, u_id:{}".format(x_idx_flat, u_idx_flat))

    test_cst.check(x_bar, u_bar, *p)


if __name__ == "__main__":
    dt = 0.1
    N = 10
    # robot mdl
    robot = MobileManipulator3D()
    motion_cst = MotionConstraint(dt, N, robot)

    q = [0.0, 0.0, 0.0] + [0.0, -2.3562, -1.5708, -2.3562, -1.5708, 1.5708]
    v = np.zeros(9)
    x = np.hstack((np.array(q), v))
    x_bar = np.tile(x, (N + 1, 1))
    u_bar = np.zeros((N, 9))
    xo = x.copy()

    # sym mdl
    params = [xo]
    A_sym, b_sym = motion_cst.linearize(x_bar, u_bar, *params)

    Ad = np.eye(18)
    Ad[:9, 9:] = np.eye(9) * dt
    Bd = np.zeros((18, 9))
    Bd[:9] = np.eye(9) * 0.5 * dt * dt
    Bd[9:] = np.eye(9) * dt

    # num mdl (double integrator)
    A_num = np.eye((N+1)*18)
    F = np.zeros_like(A_num)
    F[18:, :-18] = block_diag(*([Ad]*N))
    A_num = A_num - F

    B_num = -block_diag(*([Bd]*N))
    B_num = np.vstack((np.zeros((18, 9*N)), B_num))
    A_num = np.hstack((A_num, B_num))
    A_diff = np.linalg.norm(A_sym - A_num)
    b_diff = np.linalg.norm(b_sym)

    print("Testing Motion Model Linearization fmdl approx A dz + b")
    print("Diff A: {}".format(A_diff))
    print("Diff b: {}".format(b_diff))
    print(motion_cst.check(x_bar, u_bar, *params))

    testNonlinearConstraint()
    testHiearchicalConstraint()
    testBoxConstraint()
