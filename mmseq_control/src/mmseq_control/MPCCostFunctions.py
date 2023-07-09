from abc import ABC, abstractmethod
import timeit

import numpy as np
from scipy.linalg import block_diag
import casadi as cs
from mmseq_control.robot import MobileManipulator3D

class RBF:
    mu_sym = cs.MX.sym('mu')
    zeta_sym = cs.MX.sym('zeta')
    h_sym = cs.MX.sym('h')
    s_sym = cs.MX.sym('s')

    B_eqn_list = [-mu_sym * cs.log(h_sym),
                  mu_sym * (0.5 * (((h_sym - 2 * zeta_sym) / zeta_sym) ** 2 - 1) - cs.log(zeta_sym))]
    B_eqn = cs.conditional(s_sym, B_eqn_list, 0, False)
    B_fcn = cs.Function("B_fcn", [s_sym, h_sym, mu_sym, zeta_sym], [B_eqn])

    B_hess_eqn, B_grad_eqn = cs.hessian(B_eqn, h_sym)
    B_hess_fcn = cs.Function("ddBddh_fcn", [s_sym, h_sym, mu_sym, zeta_sym], [B_hess_eqn])
    B_grad_fcn = cs.Function("dBdh_fcn", [s_sym, h_sym, mu_sym, zeta_sym], [B_grad_eqn])

class CostFunctions(ABC):
    def __init__(self, dt, nx, nu, N):
        """ MPC cost functions base class

        :param dt: discretization time step
        :param nx: state dim
        :param nu: control dim
        :param N:  prediction window
        """


        self.dt = dt
        self.nx = nx
        self.nu = nu
        self.N = N

        self.QPsize = self.nx * (self.N + 1) + self.nu * self.N
        self.x_bar_sym = cs.MX.sym('x_bar', self.nx, self.N + 1)
        self.u_bar_sym = cs.MX.sym('u_bar', self.nu, self.N)
        self.z_bar_sym = cs.vertcat(cs.vec(self.x_bar_sym), cs.vec(self.u_bar_sym))

        super().__init__()

    @abstractmethod
    def evaluate(self, x_bar, u_bar, *params):
        """ evaluate cost function over the prediction window

        :param x_bar: predicted state trajectory numpy.ndarray [N+1, nx]
        :param u_bar: predicted control trajectory numpy.ndarray[N, nu]
        :param r_bar: reference trajectory numpy.ndarray [N+1, nr]
        :return: J: cost function value
        """
        pass

    @abstractmethod
    def quad(self, x_bar, u_bar, *params):
        """ quadratize(second order taylor expansion) of cost function around x_bar, u_bar, r_bar

        :param x_bar: linearization state trajectory numpy.ndarray [N+1, nx]
        :param u_bar: linearization control trajectory numpy.ndarray[N, nu]
        :param r_bar: reference trajectory numpy.ndarray [N+1, nr]
        :return: H: (approximated) hessian of the cost function, g: gradient of the cost function
                J \approx 0.5 * dz_bar^T H dz_bar + g^T dz_bar + c (ignored)
        """
        pass

    @abstractmethod
    def get_hess_fcn(self):
        pass

class LinearLeastSquare(CostFunctions):
    def __init__(self, dt, nx, nu, N, nr, params):
        """ Linear least square cost function
            J = ||A z_bar + b||^2_W
            A is a constant matrix, b is a constant vector and treated as a parameter

        :param dt: discretization time step
        :param nx: state dim
        :param nu: control dim
        :param N:  prediction window
        :param nr: output dimension
        :param params: python dictionary {A:[nr x nz] 2d numpy array, W:[nr x nr] 2d numpy array}
        """
        super().__init__(dt, nx, nu, N)
        self.nr = nr
        self.A = params["A"]
        self.W = params["W"]
        self.b_sym = cs.MX.sym("b", self.nr)
        self._setupSymMdl()

    def _setupSymMdl(self):
        y = self.A @ self.z_bar_sym + self.b_sym
        self.J_eqn = 0.5 * y.T @ self.W @ y
        self.hess_eqn, self.grad_eqn = cs.hessian(self.J_eqn, self.z_bar_sym)
        self.J_fcn = cs.Function('J', [self.x_bar_sym, self.u_bar_sym, self.b_sym], [self.J_eqn],
                                            ["x_bar", "u_bar", "b"], ["J"]).expand()
        self.grad_fcn = cs.Function('dJdz', [self.z_bar_sym, self.b_sym], [self.grad_eqn]).expand()
        self.hess_fcn = cs.Function('ddJddz', [self.z_bar_sym, self.b_sym], [self.hess_eqn]).expand()

    def evaluate(self, x_bar, u_bar, *params):
        return self.J_fcn(x_bar.T, u_bar.T, params[0].T)

    def quad(self, x_bar, u_bar, *params):
        z_bar = np.hstack((x_bar.flatten(), u_bar.flatten()))
        H = self.hess_fcn(z_bar, params[0].T)
        g = self.grad_fcn(z_bar, params[0].T)

        return H, g

    def get_hess_fcn(self):
        return self.hess_fcn


class TrackingCostFunction(CostFunctions):
    def __init__(self, dt, nx, nu, N, nr, f_fcn, params):
        """ Nonlinear least square cost function
            J = \sum_{k=0}^N ||f_fcn(x(t_k)) - r(t_k)||^2_Qk + ||f_fcn(x(t_N)) - r(t_N)||^2_P

        :param dt: discretization time step
        :param nx: state dim
        :param nu: control dim
        :param N:  prediction window
        :param nr: ref dim
        :param f_fcn: output/observation function, casadi.Function
        :param params: cost function params
        """
        super().__init__(dt, nx, nu, N)
        self.nr = nr
        self.f_fcn = f_fcn.expand()
        self.r_bar_sym = cs.MX.sym("r_bar", self.nr, self.N+1)

        self.Qk = params["Qk"]
        self.Q = block_diag(*([self.Qk] * self.N))
        self.P = params["P"]
        self.W = block_diag(self.Q, self.P)
        self._setupSymMdl()

    def _setupSymMdl(self):
        self.J_eqn, self.e_bar_eqn, self.J_bar_eqn = self._getCostSymEqn(self.f_fcn, self.x_bar_sym, self.u_bar_sym, self.r_bar_sym)
        self.hess_eqn, self.grad_eqn = cs.hessian(self.J_eqn, self.z_bar_sym)
        self.debardz_eqn = cs.jacobian(self.e_bar_eqn, self.z_bar_sym)
        self.hess_approx_eqn = self.debardz_eqn.T @ self.W @self.debardz_eqn

        self.J_fcn = cs.Function('J', [self.x_bar_sym, self.u_bar_sym, self.r_bar_sym], [self.J_eqn], ["x_bar", "u_bar", "r_bar"], ["J"]).expand()
        self.J_bar_fcn = cs.Function('J_vec', [self.x_bar_sym, self.u_bar_sym, self.r_bar_sym], [self.J_bar_eqn], ["x_bar", "u_bar", "r_bar"], ["J"]).expand()
        self.e_bar_fcn = cs.Function('e_bar', [self.x_bar_sym, self.u_bar_sym, self.r_bar_sym], [self.e_bar_eqn], ["x_bar", "u_bar", "r_bar"], ["e_bar"]).expand()
        self.grad_fcn = cs.Function('dJdz', [self.z_bar_sym, self.r_bar_sym], [self.grad_eqn]).expand()
        self.hess_fcn = cs.Function('ddJddz', [self.z_bar_sym, self.r_bar_sym], [self.hess_eqn]).expand()
        self.hess_approx_fcn = cs.Function('dJdzdJdzT', [self.z_bar_sym, self.r_bar_sym], [self.hess_approx_eqn]).expand()

    def _getCostSymEqn(self, f_fcn, x_bar_sym, u_bar_sym, r_bar_sym):
        J_list = []
        J = cs.MX.zeros(1)
        e_bar_eqn = cs.MX([])

        for k in range(self.N + 1):
            xk = x_bar_sym[:, k]
            yk = f_fcn(xk)
            rk = r_bar_sym[:, k]
            ek = yk - rk
            e_bar_eqn = cs.vertcat(e_bar_eqn, ek)

            if k < self.N:
                # J += 0.5 * ek.T @ self.Qk @ ek
                J_list.append(0.5 * ek.T @ self.Qk @ ek)
            else:
                # J += 0.5 * ek.T @ self.P @ ek
                J_list.append(0.5 * ek.T @ self.P @ ek)


        return sum(J_list)[0], e_bar_eqn, cs.vertcat(*J_list)

    def evaluate(self, x_bar, u_bar, *params):
        return self.J_fcn(x_bar.T, u_bar.T, params[0].T).toarray()[0][0]

    def quad(self, x_bar, u_bar, *params):
        z_bar = np.hstack((x_bar.flatten(), u_bar.flatten()))
        H = self.hess_approx_fcn(z_bar, params[0].T)
        g = self.grad_fcn(z_bar, params[0].T)
        return H, g

    def get_hess_fcn(self):
        return self.hess_approx_fcn


class EEPos3CostFunction(TrackingCostFunction):
    def __init__(self, dt, N, robot_mdl, params):
        self.name = "EEPos3"
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        nr = 3
        fk_ee = robot_mdl.kinSymMdls[robot_mdl.tool_link_name]
        f_fcn = cs.Function("fee", [robot_mdl.x_sym], [fk_ee(robot_mdl.q_sym)[0]])
        cost_params = {}
        cost_params["Qk"] = np.eye(nr) * params["Qk"]
        cost_params["P"] = np.eye(nr) * params["P"]
        super().__init__(dt, nx, nu, N, nr, f_fcn, cost_params)


class EEPos3BaseFrameCostFunction(TrackingCostFunction):
    def __init__(self, dt, N, robot_mdl, params):
        self.name = "EEPos3"
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        nr = 3
        fk_ee = robot_mdl._getFk(robot_mdl.tool_link_name, base_frame=True)
        f_fcn = cs.Function("fee", [robot_mdl.x_sym], [fk_ee(robot_mdl.q_sym)[0]])
        cost_params = {}
        cost_params["Qk"] = np.eye(nr) * params["Qk"]
        cost_params["P"] = np.eye(nr) * params["P"]
        super().__init__(dt, nx, nu, N, nr, f_fcn, cost_params)

class BasePos2CostFunction(TrackingCostFunction):
    def __init__(self, dt, N, robot_mdl, params):
        self.name = "BasePos2"
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        nr = 2
        fk_b = robot_mdl.kinSymMdls["base"]
        f_fcn = cs.Function("fb", [robot_mdl.x_sym], [fk_b(robot_mdl.q_sym)[0]])
        cost_params = {}
        cost_params["Qk"] = np.eye(nr) * params["Qk"]
        cost_params["P"] = np.eye(nr) * params["P"]

        super().__init__(dt, nx, nu, N, nr, f_fcn, cost_params)

class ControlEffortCostFuncitonNew(CostFunctions):
    def __init__(self, dt, N, robot_mdl, params):
        self.name = "ControlEffort"

        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        nr = nx * (N+1) + nu * N


        # x u penalizaiton
        Qq = [params["Qqb"]] * (robot_mdl.DoF - robot_mdl.numjoint) + [params["Qqa"]] * robot_mdl.numjoint
        Qv = [params["Qvb"]] * (nu - robot_mdl.numjoint) + [params["Qva"]] * robot_mdl.numjoint
        Qx = np.diag((Qq + Qv) * N + [0] * nx)

        Qu = [params["Qub"]] * (nu - robot_mdl.numjoint) + [params["Qua"]] * robot_mdl.numjoint
        Qu = np.diag(Qu * N)
        ll_params = {}
        ll_params["W"] = block_diag(Qx, Qu)
        ll_params["A"] = np.eye(nr)
        self.xu_cost = LinearLeastSquare(dt, nx, nu, N, nr, ll_params)

        # du penalization
        Qdu = [params["Qdub"]] * (nu - robot_mdl.numjoint) + [params["Qdua"]] * robot_mdl.numjoint
        Qdu = np.diag(Qdu * N)
        Lambda = np.zeros((N*nu, N*nu))
        Lambda[nu:, :-nu] = -np.eye((N-1)*nu)
        Lambda += np.eye(N*nu)

        ll_params_du = {}
        ll_params_du["W"] = Qdu
        ll_params_du["A"] = np.hstack((np.zeros((N*nu, (N+1)*nx)), Lambda))
        self.du_cost = LinearLeastSquare(dt, nx, nu, N, nu*N, ll_params_du)

        super().__init__(dt, nx, nu, N)

    def evaluate(self, x_bar, u_bar, *params):
        Jxu = self.xu_cost.evaluate(x_bar, u_bar, np.zeros(self.QPsize))
        Jdu = self.du_cost.evaluate(x_bar, u_bar, np.hstack((*params, np.zeros((self.N-1)*self.nu))))
        return Jxu + Jdu

    def quad(self, x_bar, u_bar, *params):
        Hxu, gxu = self.xu_cost.quad(x_bar, u_bar, np.zeros(self.QPsize))
        Hdu, gdu = self.du_cost.quad(x_bar, u_bar, np.hstack((*params, np.zeros((self.N-1)*self.nu))))
        return Hxu + Hdu, gxu + gdu

    def get_hess_fcn(self):
        pass

class ControlEffortCostFunciton(LinearLeastSquare):
    def __init__(self, dt, N, robot_mdl, params):
        self.name = "ControlEffort"

        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        nr = nx * (N+1) + nu * N


        # x u penalizaiton
        Qq = [params["Qqb"]] * (robot_mdl.DoF - robot_mdl.numjoint) + [params["Qqa"]] * robot_mdl.numjoint
        Qv = [params["Qvb"]] * (nu - robot_mdl.numjoint) + [params["Qva"]] * robot_mdl.numjoint
        Qx = np.diag((Qq + Qv) * N + [0] * nx)

        Qu = [params["Qub"]] * (nu - robot_mdl.numjoint) + [params["Qua"]] * robot_mdl.numjoint
        Qu = np.diag(Qu * N)
        ll_params = {}
        ll_params["W"] = block_diag(Qx, Qu)
        ll_params["A"] = np.eye(nr)
        super().__init__(dt, nx, nu, N, nr, ll_params)

    def evaluate(self, x_bar, u_bar, *params):
        return super().evaluate(x_bar, u_bar, *[np.zeros(self.QPsize)])

    def quad(self, x_bar, u_bar, *params):
        return super().quad(x_bar, u_bar, *[np.zeros(self.QPsize)])

class SoftConstraintsRBFCostFunction(CostFunctions):
    def __init__(self, mu, zeta, cst_obj, name="SoftConstraint"):
        super().__init__(cst_obj.dt, cst_obj.nx, cst_obj.nu, cst_obj.N)

        self.name = name
        self.mu = mu
        self.zeta = zeta

        self.params_sym = cst_obj.params_sym
        self.params_name = cst_obj.params_name

        self.h_eqn = -cst_obj.g_fcn(self.x_bar_sym, self.u_bar_sym, *self.params_sym)
        self.dhdz_eqn = -cst_obj.grad_fcn(self.z_bar_sym, *self.params_sym)
        self.nh = self.h_eqn.shape[0]
        self.s_sym = cs.MX.sym("s_"+self.name, self.nh)



        J_eqn_list = [RBF.B_fcn(self.s_sym[k], self.h_eqn[k], self.mu, self.zeta) for k in range(self.nh)]
        self.J_eqn = sum(J_eqn_list)
        self.hess_eqn, self.grad_eqn = cs.hessian(self.J_eqn, self.z_bar_sym)
        ddBddh_eqn_list = [RBF.B_hess_fcn(self.s_sym[k], self.h_eqn[k], self.mu, self.zeta) for k in range(self.nh)]
        ddBddh_eqn = cs.diag(cs.vertcat(*ddBddh_eqn_list))
        self.hess_approx_eqn = self.dhdz_eqn.T @ ddBddh_eqn @ self.dhdz_eqn

        self.h_fcn = cs.Function("h_"+self.name, [self.x_bar_sym, self.u_bar_sym, *self.params_sym], [self.h_eqn]).expand()
        self.J_fcn = cs.Function("J_" + self.name, [self.x_bar_sym, self.u_bar_sym, *self.params_sym, self.s_sym], [self.J_eqn]).expand()
        self.hess_fcn = cs.Function("ddJddz_"+self.name, [self.z_bar_sym, *self.params_sym, self.s_sym], [self.hess_eqn]).expand()
        self.hess_approx_fcn = cs.Function("ddJddz_approx_" + self.name, [self.z_bar_sym, *self.params_sym, self.s_sym],
                                    [self.hess_approx_eqn]).expand()
        self.grad_fcn = cs.Function("dJdz_" + self.name, [self.z_bar_sym, *self.params_sym, self.s_sym],
                                    [self.grad_eqn]).expand()

    def evaluate(self, x_bar, u_bar, *params):
        s = self.h_fcn(x_bar.T, u_bar.T, *params) < self.zeta

        return self.J_fcn(x_bar.T, u_bar.T, *params, s)

    def quad(self, x_bar, u_bar, *params):
        s = self.h_fcn(x_bar.T, u_bar.T, *params) < self.zeta
        z_bar = np.hstack((x_bar.flatten(), u_bar.flatten()))
        H = self.hess_approx_fcn(z_bar, *params, s)
        g = self.grad_fcn(z_bar, *params, s)
        return H, g

    def get_hess_fcn(self):
        return self.hess_approx_fcn

class SumOfCostFunctions(CostFunctions):

    def __init__(self, cost_fcn_obj):
        self.name = "Sum"
        self.cost_fcn_obj = cost_fcn_obj
        J_fcns = [c.J_fcn for c in cost_fcn_obj]
        self.J_fcn = self._sum_cs_fcns(J_fcns)
        self.hess_fcn = self._sum_cs_fcns([c.get_hess_fcn() for c in cost_fcn_obj], n_comm_in=1)
        self.grad_fcn = self._sum_cs_fcns([c.grad_fcn for c in cost_fcn_obj], n_comm_in=1)

    def evaluate(self, x_bar, u_bar, *params):
        params = [p.T for p in params]
        return self.J_fcn(x_bar.T, u_bar.T, *params)

    def quad(self, x_bar, u_bar, *params):
        z_bar = np.hstack((x_bar.flatten(), u_bar.flatten()))
        params = [p.T for p in params]

        H = self.hess_fcn(z_bar, *params)
        g = self.grad_fcn(z_bar, *params)

        return H, g

    def _sum_cs_fcns(self, fcns, n_comm_in=2):
        common_input_sysms = fcns[0].mx_in()[:n_comm_in]
        param_syms = []
        output_syms = []
        for f in fcns:
            f_params = f.mx_in()[n_comm_in:]
            f_in = common_input_sysms + f_params
            f_out = f(*f_in)
            output_syms.append(f_out)
            param_syms += f_params

        sum_fcn = cs.Function('sum', common_input_sysms + param_syms, [sum(output_syms)])

        return sum_fcn

    def get_hess_fcn(self):
        return self.hess_fcn

def time_quad():
    setup = """
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import block_diag
from model.robot import MobileManipulator3D
from ctrl.MPCCostFunctions import BasePos2CostFunction, EEPos3CostFunction,ControlEffortCostFunciton
import casadi as cs
dt = 0.1
N = 10
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
cost_eff = ControlEffortCostFunciton(dt, N, robot, cost_params["effort"])
cost_fcn = cost_ee

q = [0.0, 0.0, 0.0] + [0.0, -2.3562, -1.5708, -2.3562, -1.5708, 1.5708]
v = np.zeros(9)
x = np.hstack((np.array(q), v))
x_bar = np.tile(x, (N + 1, 1))
u_bar = np.zeros((N, 9))
r_bar = np.array([1] * cost_fcn.nr)
r_bar = np.tile(r_bar, (N + 1, 1))
    """

    print("Normal call {}".format(timeit.timeit(
        "cost_fcn.quad(x_bar, u_bar, r_bar)", setup=setup, number=2)))

if __name__ == "__main__":
    dt = 0.1
    N = 10
    # robot mdl
    from mmseq_utils import parsing

    config = parsing.load_config(
        "/home/tracy/Projects/mm_catkin_ws/src/mm_sequential_tasks/mmseq_run/config/simple_experiment.yaml")
    robot = MobileManipulator3D(config["controller"])

    cost_base = BasePos2CostFunction(dt, N, robot, config["controller"]["cost_params"]["BasePos2"])
    cost_ee = EEPos3CostFunction(dt, N, robot, config["controller"]["cost_params"]["EEPos3"])
    cost_eff = ControlEffortCostFunciton(dt, N, robot, config["controller"]["cost_params"]["Effort"])
    cost_fcn = cost_ee

    q = [0.0, 0.0, 0.0] + [0.0, -2.3562, -1.5708, -2.3562, -1.5708, 1.5708]
    v = np.zeros(9)
    x = np.hstack((np.array(q), v))
    QPsize = 18 * (N+1) + 9 * N
    x_bar = np.tile(x, (N+1, 1))
    u_bar = np.zeros((N, 9))
    r_bar = np.array([1] * cost_fcn.nr)
    r_bar = np.tile(r_bar, (N+1, 1))
    # r_bar = np.ones(9)

    J_sym = cost_fcn.evaluate(x_bar, u_bar, r_bar)
    H_sym, g_sym = cost_fcn.quad(x_bar, u_bar, r_bar)
    g_num = np.zeros(cost_fcn.QPsize)

    eps = 1e-7

    for i in range(x_bar.shape[0]):
        for j in range(x_bar.shape[1]):
            x_bar_p = x_bar.copy()
            x_bar_p[i][j] += eps
            J_p = cost_fcn.evaluate(x_bar_p, u_bar, r_bar)
            g_num[i*x_bar.shape[1] + j] = (J_p - J_sym) / eps

    for i in range(u_bar.shape[0]):
        for j in range(u_bar.shape[1]):
            u_bar_p = u_bar.copy()
            u_bar_p[i][j] += eps
            J_p = cost_fcn.evaluate(x_bar, u_bar_p, r_bar)
            indx = (N+1) * 18 + i*u_bar.shape[1] + j
            g_num[indx] = (J_p - J_sym) / eps

    print("Difference in gradient:{}".format(np.linalg.norm(g_num - g_sym)))
    print(H_sym)

    print("------------ Testing SumOfCostFunctions ---------------")

    ree_bar = np.array([1] * 3)
    ree_bar = np.tile(ree_bar, (N + 1, 1))
    rbase_bar = np.array([1] * 2)
    rbase_bar = np.tile(rbase_bar, (N + 1, 1))

    cost_fcns = [cost_ee, cost_base, cost_eff]
    params = [ree_bar, rbase_bar, np.zeros(QPsize)]
    sum_cost_functions = SumOfCostFunctions(cost_fcns)
    J_sum = sum_cost_functions.evaluate(x_bar, u_bar, *params)
    J_sum_baseline = [cost_fcns[k].evaluate(x_bar, u_bar, params[k]) for k in range(3)]
    print("J diff:{}".format(J_sum - sum(J_sum_baseline)))

    H_sum, g_sum = sum_cost_functions.quad(x_bar, u_bar, *[ree_bar, rbase_bar, np.zeros(QPsize)])
    H_sum_baseline = np.zeros((QPsize, QPsize))
    g_sum_baseline = np.zeros(QPsize)
    for k in range(3):
        H,g = cost_fcns[k].quad(x_bar, u_bar, params[k])
        H_sum_baseline += H
        g_sum_baseline += g

    print("H diff: {}, g_diff {}".format(np.linalg.norm(H_sum - H_sum_baseline), np.linalg.norm(g_sum - g_sum_baseline)))
    # time_quad()

