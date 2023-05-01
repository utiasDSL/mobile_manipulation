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
        """ nonlinear inequality constraint
                            g(x_bar, u_bar, *params) < 0

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
    def __init__(self, dt, robot_mdl, N, tol=1e-2):
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
        super().__init__(dt, nx, nu, ng, N, g_fcn, [], name, tol=tol)

class StateControlBoxConstraintNew(NonlinearConstraint):
    def __init__(self, dt, robot_mdl, N, tol=1e-2):
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        name = "xuboxnew"
        Constraint.__init__(self, dt, nx, nu, N, name)

        # state and input bounds
        ng = self.z_bar_sym.size(1)
        self.ub = np.array(ss_mdl["ub_x"] * (N+1) + ss_mdl["ub_u"] * N)
        self.lb = np.array(ss_mdl["lb_x"] * (N+1) + ss_mdl["lb_u"] * N)
        g_eqn = cs.vertcat(self.z_bar_sym, -self.z_bar_sym) - np.expand_dims(np.hstack((self.ub, -self.lb)),-1)
        # g_fcn = cs.Function('g_'+name, [self.x_bar_sym, self.u_bar_sym], [g_eqn], ["x_bar", "u_bar"], ['g_'+name])

        # input rate bounds
        # convert to delt u (i.e. time difference value)
        self.ub_du = np.array(ss_mdl["ub_udot"] * N) * dt
        self.lb_du = np.array(ss_mdl["lb_udot"] * N) * dt
        self.u_prev = cs.MX.sym("u_prev", nu)
        du_eqn = cs.vec(self.u_bar_sym) - cs.vertcat(self.u_prev, cs.vec(self.u_bar_sym[:, :-1]))
        g_eqn_du = cs.vertcat(du_eqn, -du_eqn) - np.expand_dims(np.hstack((self.ub_du, -self.lb_du)), -1)

        g_fcn = cs.Function('g_' + name, [self.x_bar_sym, self.u_bar_sym, self.u_prev], [cs.vertcat(g_eqn, g_eqn_du)], ["x_bar", "u_bar", "u_prev"], ['g_' + name])

        super().__init__(dt, nx, nu, ng, N, g_fcn, [self.u_prev], name, tol=tol)

class SignedDistanceCollisionConstraint(NonlinearConstraint):
    def __init__(self, robot_mdl, signed_distance_fcn, dt, N, d_safe, name="obstacle", tol=1e-2):
        """ Signed Distance Collision Constraint
                   - (sd(x_k) - d_safe) < 0

        :param robot_mdl: class mmseq_control.robot.MobileManipulator3D
        :param signed_distance_fcn: signed distance model between (multiple) body pair(s), casadi function
        :param dt: discretization time step
        :param N: prediction window size
        :param d_safe: safe clearance, scalar, same for all body pairs
        :param name: name of this set of collision pairs
        :param tol: tolerence for this constraint
        """
        nx = robot_mdl.ssSymMdl["nx"]
        nu = robot_mdl.ssSymMdl["nu"]
        nq = robot_mdl.q_sym.size()[0]

        Constraint.__init__(self, dt, nx, nu, N, name+"_collision")

        g_sym_list = [- signed_distance_fcn(self.x_bar_sym[:nq][k]) + d_safe for k in range(N+1)]
        g_sym = cs.vertcat(*g_sym_list)
        ng = g_sym.size()[0]

        g_fcn = cs.Function("f_"+self.name, [self.x_bar_sym, self.u_bar_sym], [g_sym], ["x_bar", "u_bar"], ["g_collision"])
        super().__init__(dt, nx, nu, nq, N, g_fcn, [], self.name)

def testCollisionConstraint(config):
    from mmseq_control.robot import  CasadiModelInterface
    from mmseq_control.MPCCostFunctions import SoftConstraintsRBFCostFunction
    casadi_model_interface = CasadiModelInterface(config["controller"])
    dt = 0.1
    N = 2
    robot_mdl = casadi_model_interface.robot
    sd_fcn = casadi_model_interface.getSignedDistanceSymMdls("self")
    const = SignedDistanceCollisionConstraint(robot_mdl, sd_fcn, dt, N, 0.1, "self")
    nx = robot_mdl.ssSymMdl["nx"]
    nu = robot_mdl.ssSymMdl["nu"]
    x_bar = np.ones((N + 1, nx)) * 0.0
    u_bar = np.ones((N, nu)) * 0

    mu = config["controller"]["collision_soft"]["mu"]
    zeta = config["controller"]["collision_soft"]["zeta"]
    const_soft = SoftConstraintsRBFCostFunction(mu, zeta, const, "SelfCollisionSoftConstraint")
    J_soft = const_soft.evaluate(x_bar, u_bar)
    print(J_soft)


def testSoftConstraint(config):
    from mmseq_control.MPCCostFunctions import SoftConstraintsRBFCostFunction
    dt = 0.1
    N = 10
    # robot mdl
    robot = MobileManipulator3D(config["controller"])
    nx = robot.ssSymMdl["nx"]
    nu = robot.ssSymMdl["nu"]
    Qpsize = (N + 1) * nx + N * nu
    x_bar = np.ones((N + 1, nx))*0.1
    u_bar = np.ones((N, nu)) * 0
    z_bar = np.hstack((x_bar.flatten(), u_bar.flatten()))
    e_bar = np.ones((N+1) * 2) * 2
    r_bar = np.ones((N+1, 2))

    # cost function params
    cost_params = {}
    cost_params["EE"] = {"Qk": 1., "P": 100.}
    cost_params["base"] = {"Qk": 1., "P": 100.}
    cost_params["effort"] = {"Qqa": 0., "Qqb": 0.,
                             "Qva": 1e-2, "Qvb": 2e-2,
                             "Qua": 1e-1, "Qub": 1e-1}

    cost_base = BasePos2CostFunction(dt, N, robot, cost_params["base"])
    cost_ee = EEPos3CostFunction(dt, N, robot, cost_params["EE"])
    base_h_cst = HierarchicalTrackingConstraint(cost_base, "base")
    # ee_h_cst = HierarchicalTrackingConstraint(cost_ee, "ee")
    xu_cst = StateControlBoxConstraint(dt, robot, N)
    mu = 0.005
    zeta = 0.0025
    base_h_cst_soft = SoftConstraintsRBFCostFunction(mu, zeta, base_h_cst, "base_hier")
    J_soft = base_h_cst_soft.evaluate(x_bar, u_bar, *[r_bar.T, e_bar.T])
    print(base_h_cst_soft.h_fcn(x_bar.T, u_bar.T, *[r_bar.T, e_bar.T]))
    print(J_soft)

    xu_cst_soft = SoftConstraintsRBFCostFunction(mu, zeta, xu_cst, "xu")
    J_soft = xu_cst_soft.evaluate(x_bar, u_bar)
    print(xu_cst_soft.h_fcn(x_bar.T, u_bar.T))
    print(J_soft)

    xu_cst_new = StateControlBoxConstraintNew(dt, robot, N)
    xu_cst_soft = SoftConstraintsRBFCostFunction(mu, zeta, xu_cst_new, "xu")

    u_prev = np.hstack((np.ones(2)*0., np.zeros(7)))
    J_soft = xu_cst_soft.evaluate(x_bar, u_bar, u_prev)
    print(xu_cst_soft.h_fcn(x_bar.T, u_bar.T, u_prev))
    print(J_soft)

def testBoxConstraint(config):
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

def testBoxConstraintNew(config):
    print("Testing Box Constraint")
    dt = 0.1
    N = 2
    # robot mdl
    robot = MobileManipulator3D(config["controller"])
    nx = robot.ssSymMdl["nx"]
    nu = robot.ssSymMdl["nu"]
    Qpsize = (N+1)*nx + N * nu
    cst = StateControlBoxConstraintNew(dt, robot, N)
    x_bar = np.ones((N + 1, nx))
    u_bar = np.ones((N, nu))
    z_bar = np.hstack((x_bar.flatten(), u_bar.flatten()))
    u_prev = -1 * np.ones(nu)

    lb = np.array(robot.ssSymMdl["lb_x"] * (N+1) + robot.ssSymMdl["lb_u"] * N)
    ub = np.array(robot.ssSymMdl["ub_x"] * (N+1) + robot.ssSymMdl["ub_u"] * N)
    Csym, dsym = cst.linearize(x_bar, u_bar, u_prev)

    Cdu = np.zeros((N*nu, Qpsize))
    Cdu[:, (N+1)*nx:] = np.eye(N * nu)
    Cdu[nu:, (N+1)*nx:-nu] = -np.eye((N-1)*nu)
    lb_du = np.array(robot.ssSymMdl["lb_udot"] * N) * dt
    ub_du = np.array(robot.ssSymMdl["ub_udot"] * N) * dt
    ub_du[:nu] += u_prev
    lb_du[:nu] += u_prev
    Cnum = np.vstack((np.eye(Qpsize), -np.eye(Qpsize), Cdu, -Cdu))
    dnum = np.hstack((-ub, lb, -ub_du, lb_du)) + Cnum @ z_bar

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
    from mmseq_utils import parsing
    config = parsing.load_config("/home/tracy/Projects/mm_catkin_ws/src/mm_sequential_tasks/mmseq_run/config/simple_experiment.yaml")
    robot = MobileManipulator3D(config["controller"])
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

    # testNonlinearConstraint()
    # testHiearchicalConstraint()
    # testBoxConstraintNew(config)
    # testSoftConstraint(config)
    testCollisionConstraint(config)