from abc import ABC, abstractmethod

import casadi as cs

from mm_utils.casadi_struct import casadi_sym_struct


class Constraint(ABC):
    def __init__(self, nx, nu, name):
        """MPC constraints base class.

        Args:
            nx (int): State dimension.
            nu (int): Control dimension.
            name (str): Name to identify the constraint.
        """

        self.nx = nx
        self.nu = nu
        self.name = name

        self.x_sym = cs.MX.sym("x", nx)
        self.u_sym = cs.MX.sym("u", nu)

        self.p_dict = None
        self.p_sym = None

        self.slack_enabled = False

        super().__init__()

    @abstractmethod
    def check(self, x, u, p):
        """Check constraint satisfaction.

        Args:
            x (ndarray): State vector.
            u (ndarray): Control input vector.
            p (ndarray): Parameter vector.

        Returns:
            ndarray: Constraint values (should be < 0 for satisfaction).
        """
        pass

    def get_p_dict(self, sym=True):
        """Get parameter dictionary with name suffixes.

        Args:
            sym (bool): If True, return symbolic parameters; if False, return zero matrices.

        Returns:
            dict or None: Parameter dictionary with keys suffixed by constraint name, or None if no parameters.
        """
        if self.p_dict is None:
            return None
        else:
            if sym:
                return {
                    key + f"_{self.name}": val for (key, val) in self.p_dict.items()
                }
            else:
                return {
                    key + f"_{self.name}": cs.DM.zeros(val.shape)
                    for (key, val) in self.p_dict.items()
                }

    def get_p_dict_default(self):
        """Get default parameter dictionary (zero values).

        Returns:
            dict or None: Default parameter dictionary with zero values, or None if no parameters.
        """
        p_dict = self.get_p_dict(False)
        return p_dict


class NonlinearConstraint(Constraint):
    def __init__(self, nx, nu, ng, g_fcn, p_dict, constraint_name):
        """Nonlinear inequality constraint g(x, u, p) < 0.

        Args:
            nx (int): State dimension.
            nu (int): Control dimension.
            ng (int): Constraint dimension.
            g_fcn (casadi.Function): Function g_fcn(x_bar_sym, u_bar_sym, *params_sym).
            p_dict (dict): Parameter dictionary.
            constraint_name (str): Name to identify the constraint.
        """

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
        """Check nonlinear constraint satisfaction.

        Args:
            x (ndarray): State vector.
            u (ndarray): Control input vector.
            p (ndarray): Parameter vector.

        Returns:
            ndarray: Constraint values (should be < 0 for satisfaction).
        """
        g = self.g_fcn(x, u, p)

        return g


class SignedDistanceConstraint(NonlinearConstraint):
    def __init__(self, robot_mdl, signed_distance_fcn, d_safe, name="obstacle"):
        """Signed Distance Constraint: -(sd(x_k, param1, param2, ...) - d_safe) < 0.

        Args:
            robot_mdl (MobileManipulator3D): Robot model.
            signed_distance_fcn (casadi.Function): Signed distance model.
            d_safe (float): Safe clearance, scalar, same for all body pairs.
            name (str): Name of this constraint.
        """
        nx = robot_mdl.ssSymMdl["nx"]
        nu = robot_mdl.ssSymMdl["nu"]
        nq = robot_mdl.q_sym.size()[0]
        ng = signed_distance_fcn.size_out(0)[0]
        p_sym = signed_distance_fcn.mx_in()[1:]
        p_name = signed_distance_fcn.name_in()[1:]
        p_dict = {name: sym for (name, sym) in zip(p_name, p_sym)}

        super().__init__(nx, nu, ng, None, p_dict, name)

        self.g_eqn = (
            -signed_distance_fcn(
                self.x_sym[:nq], *[self.p_struct[k] for k in self.p_dict.keys()]
            )
            + d_safe
        )

        self.g_fcn = cs.Function(
            "g_" + self.name, [self.x_sym, self.u_sym, self.p_sym], [self.g_eqn]
        )

        self.g_grad_eqn = cs.jacobian(self.g_eqn, cs.veccat(self.u_sym, self.x_sym))
        self.g_grad_fcn = cs.Function(
            "g_grad", [self.x_sym, self.u_sym, self.p_sym], [self.g_grad_eqn]
        )

        self.slack_enabled = True


class StateBoxConstraints(NonlinearConstraint):
    def __init__(self, robot_mdl, name="state"):
        """State Box Constraint: lb_x < x < ub_x.

        Args:
            robot_mdl (MobileManipulator3D): Robot model.
            name (str): Name of this constraint.
        """
        nx = robot_mdl.ssSymMdl["nx"]
        nu = robot_mdl.ssSymMdl["nu"]
        ng = nx * 2
        p_dict = {}
        super().__init__(nx, nu, ng, None, p_dict, name)

        self.g_eqn = cs.vertcat(
            self.x_sym - robot_mdl.ssSymMdl["ub_x"],
            robot_mdl.ssSymMdl["lb_x"] - self.x_sym,
        )
        self.g_fcn = cs.Function(
            "g_" + self.name, [self.x_sym, self.u_sym, self.p_sym], [self.g_eqn]
        )


class ControlBoxConstraints(NonlinearConstraint):
    def __init__(self, robot_mdl, name="control"):
        """Control Box Constraint: lb_u < u < ub_u.

        Args:
            robot_mdl (MobileManipulator3D): Robot model.
            name (str): Name of this constraint.
        """
        nx = robot_mdl.ssSymMdl["nx"]
        nu = robot_mdl.ssSymMdl["nu"]
        ng = nu * 2
        p_dict = {}
        super().__init__(nx, nu, ng, None, p_dict, name)

        self.g_eqn = cs.vertcat(
            self.u_sym - robot_mdl.ssSymMdl["ub_u"],
            robot_mdl.ssSymMdl["lb_u"] - self.u_sym,
        )
        self.g_fcn = cs.Function(
            "g_" + self.name, [self.x_sym, self.u_sym, self.p_sym], [self.g_eqn]
        )
