from abc import ABC

import casadi as cs
from liecasadi import SO3

from mm_control.robot import MobileManipulator3D
from mm_utils.casadi_struct import casadi_sym_struct
from mm_utils.math import casadi_SO2, casadi_SO3_log


class RBF:
    mu_sym = cs.MX.sym("mu")
    zeta_sym = cs.MX.sym("zeta")
    h_sym = cs.MX.sym("h")

    B_eqn_list = [
        -mu_sym * cs.log(h_sym),
        mu_sym
        * (0.5 * (((h_sym - 2 * zeta_sym) / zeta_sym) ** 2 - 1) - cs.log(zeta_sym)),
    ]
    s_eqn = h_sym < zeta_sym
    B_eqn = cs.conditional(s_eqn, B_eqn_list, 0, False)
    B_fcn = cs.Function("B_fcn", [h_sym, mu_sym, zeta_sym], [B_eqn])

    B_hess_eqn, B_grad_eqn = cs.hessian(B_eqn, h_sym)
    B_hess_fcn = cs.Function("ddBddh_fcn", [h_sym, mu_sym, zeta_sym], [B_hess_eqn])
    B_grad_fcn = cs.Function("dBdh_fcn", [h_sym, mu_sym, zeta_sym], [B_grad_eqn])


class CostFunctions(ABC):
    def __init__(self, nx: int, nu: int, name: str = "MPCCost"):
        """MPC cost functions base class.

        Args:
            nx (int): State dimension.
            nu (int): Control dimension.
            name (str): Name of the cost function.
        """

        self.nx = nx
        self.nu = nu
        self.name = name

        self.x_sym = cs.MX.sym("x", nx)
        self.u_sym = cs.MX.sym("u", nu)

        self.p_sym = None
        self.p_dict = None
        self.p_struct = None
        self.J_eqn = None
        self.J_fcn = None
        self.H_approx_eqn = None
        self.H_approx_fcn = None

        super().__init__()

    def evaluate(self, x, u, p):
        """Evaluate cost function at given state, control, and parameters.

        Args:
            x (ndarray): State vector.
            u (ndarray): Control input vector.
            p (ndarray): Parameter vector.

        Returns:
            ndarray or None: Cost value, or None if cost function is not set.
        """
        if self.J_fcn is not None:
            return self.J_fcn(x, u, p).toarray().flatten()
        else:
            return None

    def get_p_dict(self):
        """Get parameter dictionary with name suffixes.

        Returns:
            dict or None: Parameter dictionary with keys suffixed by function name, or None if no parameters.
        """
        if self.p_dict is None:
            return None
        else:
            return {key + f"_{self.name}": val for (key, val) in self.p_dict.items()}

    def get_custom_H_fcn(self):
        """Get custom Hessian approximation function.

        Returns:
            casadi.Function or None: Custom Hessian approximation function, or None if not set.
        """
        return self.H_approx_fcn


# Base class for trajectory tracking costs (shared by position and velocity)
class TrajectoryTrackingCostFunction(CostFunctions):
    """Base class for trajectory tracking cost functions."""

    def __init__(self, nx: int, nu: int, nr: int, f_fcn: cs.Function, name: str):
        """Initialize trajectory tracking cost function.

        Args:
            nx (int): State dimension.
            nu (int): Control dimension.
            nr (int): Reference dimension.
            f_fcn (casadi.Function): Function that maps state to tracked quantity.
            name (str): Name of the cost function.
        """
        super().__init__(nx, nu, name)
        self.nr = nr
        self.f_fcn = f_fcn.expand()
        self.p_dict = {
            "r": cs.MX.sym("r_" + self.name, self.nr),
            "W": cs.MX.sym("W_" + self.name, self.nr, self.nr),
        }
        self.p_struct = casadi_sym_struct(self.p_dict)
        self.p_sym = self.p_struct.cat
        self.W = self.p_struct["W"]
        self.r = self.p_struct["r"]

        # Setup cost function: J = 1/2 ||f(x) - r||^2_W
        y = self.f_fcn(self.x_sym)
        e = y - self.r
        self.e_eqn = e
        self.J_eqn = 0.5 * e.T @ self.W @ e
        self.J_fcn = cs.Function(
            "J_" + self.name,
            [self.x_sym, self.u_sym, self.p_sym],
            [self.J_eqn],
            ["x", "u", "r"],
            ["J"],
        ).expand()
        self.e_fcn = cs.Function(
            "e_" + self.name,
            [self.x_sym, self.u_sym, self.r],
            [self.e_eqn],
            ["x", "u", "r"],
            ["e"],
        ).expand()
        dedx = cs.jacobian(e, self.x_sym)
        self.H_approx_eqn = cs.diagcat(
            cs.MX.zeros(self.nu, self.nu), dedx.T @ self.W @ dedx
        )
        self.H_approx_fcn = cs.Function(
            "H_approx_" + self.name,
            [self.x_sym, self.u_sym, self.p_sym],
            [self.H_approx_eqn],
            ["x", "u", "r"],
            ["H_approx"],
        ).expand()

    def get_e(self, x, u, r):
        """Get tracking error vector.

        Args:
            x (ndarray): State vector.
            u (ndarray): Control input vector.
            r (ndarray): Reference vector.

        Returns:
            ndarray: Tracking error vector.
        """
        return self.e_fcn(x, u, r).toarray().flatten()


class VelocityCostFunction(TrajectoryTrackingCostFunction):
    """Velocity tracking cost function for end-effector or base."""

    def __init__(self, robot_mdl, params, target="EE", dimension=3, name=None):
        """Initialize velocity cost function.

        Args:
            robot_mdl (MobileManipulator3D): Robot model.
            params (dict): Cost function parameters.
            target (str): Target to track, either "EE" or "base".
            dimension (int): Dimension of velocity to track (3 or 6).
            name (str, optional): Name of the cost function. If None, auto-generated.
        """
        ss_mdl = robot_mdl.ssSymMdl
        nx, nu, nr = ss_mdl["nx"], ss_mdl["nu"], dimension

        if target == "EE":
            if dimension == 6:
                # Use 6D spatial Jacobian (linear + angular velocity)
                jac_key = robot_mdl.tool_link_name + "_spatial"
                if jac_key in robot_mdl.jacSymMdls:
                    jac = robot_mdl.jacSymMdls[jac_key]
                else:
                    # Fallback to 3D if spatial Jacobian not available
                    jac = robot_mdl.jacSymMdls[robot_mdl.tool_link_name]
                    # Pad with zeros for angular velocity (not ideal but works)
                    jac_3d = jac(robot_mdl.q_sym)
                    jac_6d = cs.vertcat(jac_3d, cs.MX.zeros(3, jac_3d.shape[1]))
                    jac = cs.Function("jac_6d", [robot_mdl.q_sym], [jac_6d])
                f_fcn = cs.Function(
                    "vee", [robot_mdl.x_sym], [jac(robot_mdl.q_sym) @ robot_mdl.v_sym]
                )
            else:
                # Use 3D position Jacobian (linear velocity only)
                jac = robot_mdl.jacSymMdls[robot_mdl.tool_link_name]
                f_fcn = cs.Function(
                    "vee", [robot_mdl.x_sym], [jac(robot_mdl.q_sym) @ robot_mdl.v_sym]
                )
        else:  # base
            if dimension == 3:
                f_fcn = cs.Function("vb", [robot_mdl.x_sym], [robot_mdl.vb_sym])
            else:
                jac = robot_mdl.jacSymMdls["base"]
                f_fcn = cs.Function(
                    "vb", [robot_mdl.x_sym], [jac(robot_mdl.q_sym) @ robot_mdl.v_sym]
                )

        if name is None:
            name = f"{target}Vel{dimension}"

        super().__init__(nx, nu, nr, f_fcn, name)


class PoseSE3CostFunction(CostFunctions):
    """End-effector pose tracking cost function in SE(3)."""

    def __init__(self, robot_mdl, params, base_frame=False, name=None):
        """Initialize SE(3) pose cost function.

        Args:
            robot_mdl (MobileManipulator3D): Robot model.
            params (dict): Cost function parameters.
            base_frame (bool): If True, express pose in base frame; if False, in world frame.
            name (str, optional): Name of the cost function. If None, auto-generated.
        """
        ss_mdl = robot_mdl.ssSymMdl
        nx, nu = ss_mdl["nx"], ss_mdl["nu"]

        fk = robot_mdl._getFk(robot_mdl.tool_link_name, base_frame=base_frame)
        p_ee, rot_ee = fk(robot_mdl.q_sym)
        f_fcn = cs.Function("fee", [robot_mdl.x_sym], [p_ee, rot_ee])

        if name is None:
            frame_str = "BaseFrame" if base_frame else ""
            name = f"EEPoseSE3{frame_str}"

        super().__init__(nx, nu, name)
        self.nr = 6
        self.p_dict = {
            "r": cs.MX.sym("r_" + self.name, self.nr),
            "W": cs.MX.sym("W_" + self.name, self.nr, self.nr),
        }
        self.p_struct = casadi_sym_struct(self.p_dict)
        self.p_sym = self.p_struct.cat
        self.W = self.p_struct["W"]
        self.r = self.p_struct["r"]

        # Setup SE3 pose cost
        r_pos = self.r[:3]
        r_rot_euler = self.r[3:]
        pos, rot = f_fcn(self.x_sym)
        e_pos = pos - r_pos
        orn = SO3.from_matrix(rot)
        rot_inv = SO3(cs.vertcat(-orn.xyzw[:3], orn.xyzw[3])).as_matrix()
        r_rot = SO3.from_euler(r_rot_euler).as_matrix()
        e_rot = casadi_SO3_log(rot_inv @ r_rot)

        self.e_eqn = cs.vertcat(e_pos, e_rot)
        self.J_eqn = 0.5 * self.e_eqn.T @ self.W @ self.e_eqn
        self.J_fcn = cs.Function(
            "J_" + self.name,
            [self.x_sym, self.u_sym, self.p_sym],
            [self.J_eqn],
            ["x", "u", "r"],
            ["J"],
        ).expand()
        self.e_fcn = cs.Function(
            "e_" + self.name,
            [self.x_sym, self.u_sym, self.r],
            [self.e_eqn],
            ["x", "u", "r"],
            ["e"],
        ).expand()
        dedx = cs.jacobian(self.e_eqn, self.x_sym)
        self.H_approx_eqn = cs.diagcat(
            cs.MX.zeros(self.nu, self.nu), dedx.T @ self.W @ dedx
        )
        self.H_approx_fcn = cs.Function(
            "H_approx_" + self.name,
            [self.x_sym, self.u_sym, self.p_sym],
            [self.H_approx_eqn],
            ["x", "u", "r"],
            ["H_approx"],
        ).expand()

    def get_e(self, x, u, r):
        """Get tracking error vector.

        Args:
            x (ndarray): State vector.
            u (ndarray): Control input vector.
            r (ndarray): Reference vector.

        Returns:
            ndarray: Tracking error vector.
        """
        return self.e_fcn(x, u, r).toarray().flatten()


class BasePoseSE2CostFunction(CostFunctions):
    """Base pose tracking cost function in SE(2)."""

    def __init__(self, robot_mdl, params):
        """Initialize SE(2) base pose cost function.

        Args:
            robot_mdl (MobileManipulator3D): Robot model.
            params (dict): Cost function parameters.
        """
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        super().__init__(nx, nu, "BasePoseSE2")

        self.nr = 3
        self.p_dict = {
            "r": cs.MX.sym("r_" + self.name, self.nr),
            "W": cs.MX.sym("W_" + self.name, self.nr, self.nr),
        }
        self.p_struct = casadi_sym_struct(self.p_dict)
        self.p_sym = self.p_struct.cat

        self.W = self.p_struct["W"]
        self.r = self.p_struct["r"]

        xy, h = self.x_sym[:2], self.x_sym[2]

        # position
        e_pos = xy - self.r[:2]

        # heading
        Rinv = casadi_SO2(-h)
        Rd = casadi_SO2(self.r[2])
        Rerr = Rinv @ Rd

        e_h = cs.atan2(Rerr[1, 0], Rerr[0, 0])

        self.e_eqn = cs.vertcat(e_pos, e_h)
        self.J_eqn = 0.5 * self.e_eqn.T @ self.W @ self.e_eqn

        self.J_fcn = cs.Function(
            "J_" + self.name,
            [self.x_sym, self.u_sym, self.p_sym],
            [self.J_eqn],
            ["x", "u", "r"],
            ["J"],
        ).expand()
        self.e_fcn = cs.Function(
            "e_" + self.name,
            [self.x_sym, self.u_sym, self.r],
            [self.e_eqn],
            ["x", "u", "r"],
            ["e"],
        ).expand()

        dedx = cs.jacobian(self.e_eqn, self.x_sym)
        self.H_approx_eqn = cs.diagcat(
            cs.MX.zeros(self.nu, self.nu), dedx.T @ self.W @ dedx
        )
        self.H_approx_fcn = cs.Function(
            "H_approx_" + self.name,
            [self.x_sym, self.u_sym, self.p_sym],
            [self.H_approx_eqn],
            ["x", "u", "r"],
            ["H_approx"],
        ).expand()

    def get_e(self, x, u, r):
        """Get tracking error vector.

        Args:
            x (ndarray): State vector.
            u (ndarray): Control input vector.
            r (ndarray): Reference vector.

        Returns:
            ndarray: Tracking error vector.
        """
        return self.e_fcn(x, u, r).toarray().flatten()


class ControlEffortCostFunction(CostFunctions):
    """Control effort cost function (quadratic in state and control)."""

    def __init__(self, robot_mdl, params):
        """Initialize control effort cost function.

        Args:
            robot_mdl (MobileManipulator3D): Robot model.
            params (dict): Cost function parameters with weight matrices.
        """
        ss_mdl = robot_mdl.ssSymMdl
        self.params = params
        super().__init__(ss_mdl["nx"], ss_mdl["nu"], "ControlEffort")

        self.p_dict = {
            "Qqb": cs.MX.sym("Qqb_" + self.name, 3),
            "Qqa": cs.MX.sym("Qqa_" + self.name, 6),
            "Qvb": cs.MX.sym("Qvb_" + self.name, 3),
            "Qva": cs.MX.sym("Qva_" + self.name, 6),
            "Qub": cs.MX.sym("Qub_" + self.name, 3),
            "Qua": cs.MX.sym("Qua_" + self.name, 6),
        }
        self.p_struct = casadi_sym_struct(self.p_dict)
        self.p_sym = self.p_struct.cat

        self._setupSymMdl()

    def _setupSymMdl(self):
        """Setup symbolic model for control effort cost."""
        Qq = cs.vertcat(
            self.p_struct["Qqb"],
            self.p_struct["Qqa"],
            self.p_struct["Qvb"],
            self.p_struct["Qva"],
        )
        Qx = cs.diag(Qq)

        Qu = cs.vertcat(self.p_struct["Qub"], self.p_struct["Qua"])
        Qu = cs.diag(Qu)

        self.J_eqn = (
            0.5 * self.x_sym.T @ Qx @ self.x_sym + 0.5 * self.u_sym.T @ Qu @ self.u_sym
        )
        self.J_fcn = cs.Function(
            "J_" + self.name,
            [self.x_sym, self.u_sym, self.p_sym],
            [self.J_eqn],
            ["x", "u", "r"],
            ["J"],
        ).expand()
        self.H_approx_eqn = cs.diagcat(Qu, Qx)
        self.H_approx_fcn = cs.Function(
            "H_approx_" + self.name,
            [self.x_sym, self.u_sym, self.p_sym],
            [self.H_approx_eqn],
            ["x", "u", "r"],
            ["H_approx"],
        ).expand()


class SoftConstraintsRBFCostFunction(CostFunctions):
    """Radial basis function (RBF) cost for soft constraints."""

    def __init__(self, mu, zeta, cst_obj, name="SoftConstraint", expand=True):
        """Initialize RBF soft constraint cost function.

        Args:
            mu (float): RBF parameter mu.
            zeta (float): RBF parameter zeta.
            cst_obj (Constraint): Constraint object to soften.
            name (str): Name of the cost function.
            expand (bool): If True, expand CasADi functions for efficiency.
        """
        super().__init__(cst_obj.nx, cst_obj.nu, name)

        self.mu = mu
        self.zeta = zeta

        self.p_dict = cst_obj.get_p_dict()
        self.p_struct = casadi_sym_struct(self.p_dict)
        self.p_sym = self.p_struct.cat

        self.h_eqn = -cst_obj.g_fcn(self.x_sym, self.u_sym, self.p_sym)
        self.dhdz_eqn = -cst_obj.g_grad_fcn(self.x_sym, self.u_sym, self.p_sym)
        self.nh = self.h_eqn.shape[0]

        J_eqn_list = [
            RBF.B_fcn(self.h_eqn[k], self.mu, self.zeta) for k in range(self.nh)
        ]
        self.J_eqn = sum(J_eqn_list)
        self.J_vec_eqn = cs.vertcat(*J_eqn_list)
        ddBddh_eqn_list = [
            RBF.B_hess_fcn(self.h_eqn[k], self.mu, self.zeta) for k in range(self.nh)
        ]
        ddBddh_eqn = cs.diag(cs.vertcat(*ddBddh_eqn_list))
        self.H_approx_eqn = self.dhdz_eqn.T @ ddBddh_eqn @ self.dhdz_eqn

        self.h_fcn = cs.Function(
            "h_" + self.name, [self.x_sym, self.u_sym, self.p_sym], [self.h_eqn]
        )
        self.J_fcn = cs.Function(
            "J_" + self.name, [self.x_sym, self.u_sym, self.p_sym], [self.J_eqn]
        )
        self.J_vec_fcn = cs.Function(
            "J_vec_" + self.name, [self.x_sym, self.u_sym, self.p_sym], [self.J_vec_eqn]
        )
        self.H_approx_fcn = cs.Function(
            "H_approx_" + self.name,
            [self.x_sym, self.u_sym, self.p_sym],
            [self.H_approx_eqn],
            ["x", "u", "r"],
            ["H_approx"],
        )

        if expand:
            self.h_fcn = self.h_fcn.expand()
            self.J_fcn = self.J_fcn.expand()
            self.J_vec_fcn = self.J_vec_fcn.expand()
            self.H_approx_fcn = self.H_approx_fcn.expand()

    def evaluate_vec(self, x, u, p):
        """Evaluate cost function vector (per-constraint costs).

        Args:
            x (ndarray): State vector.
            u (ndarray): Control input vector.
            p (ndarray): Parameter vector.

        Returns:
            ndarray: Cost vector, one element per constraint.
        """
        return self.J_vec_fcn(x, u, p).toarray().flatten()

    def get_p_dict(self):
        """Get parameter dictionary.

        Returns:
            dict: Parameter dictionary.
        """
        return self.p_dict


class RegularizationCostFunction(CostFunctions):
    """Regularization cost function (quadratic in state and control)."""

    def __init__(self, nx: int, nu: int, name="Regularization"):
        """Initialize regularization cost function.

        Args:
            nx (int): State dimension.
            nu (int): Control dimension.
            name (str): Name of the cost function.
        """
        super().__init__(nx, nu, name)
        self.p_dict = {"eps": cs.MX.sym("eps_reg", 1)}
        self.p_struct = casadi_sym_struct(self.p_dict)
        self.p_sym = self.p_struct.cat
        self.J_eqn = (
            0.5
            * (self.x_sym.T @ self.x_sym + self.u_sym.T @ self.u_sym)
            * self.p_struct["eps"]
        )
        self.J_fcn = cs.Function(
            "J_" + self.name, [self.x_sym, self.u_sym, self.p_sym], [self.J_eqn]
        )
        self.H_approx_eqn = cs.MX.eye(self.nx + self.nu) * self.p_struct["eps"]
        self.H_approx_fcn = cs.Function(
            "H_approx_" + self.name,
            [self.x_sym, self.u_sym, self.p_sym],
            [self.H_approx_eqn],
            ["x", "u", "eps"],
            ["H_approx"],
        )


class ManipulabilityCostFunction(CostFunctions):
    """Manipulability cost function (encourages high manipulability)."""

    def __init__(self, robot_mdl: MobileManipulator3D, name: str = "Manipulability"):
        """Initialize manipulability cost function.

        Args:
            robot_mdl (MobileManipulator3D): Robot model.
            name (str): Name of the cost function.
        """
        ss_mdl = robot_mdl.ssSymMdl
        nx = ss_mdl["nx"]
        nu = ss_mdl["nu"]
        super().__init__(nx, nu, name)

        self.p_dict = {"w": cs.MX.sym("w", 1)}
        self.p_struct = casadi_sym_struct(self.p_dict)
        self.p_sym = self.p_struct.cat

        self.J_eqn = (
            robot_mdl.arm_manipulability_fcn(robot_mdl.q_sym) ** 2
            * self.p_dict["w"]
            * 0.5
        )
        self.J_fcn = cs.Function("fee", [robot_mdl.x_sym], [self.J_eqn])


# Cost Function Registry
class CostFunctionRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str, factory):
        """Register a cost function factory.

        Args:
            name (str): Name of the cost function.
            factory (callable): Factory function that creates the cost function.
        """
        cls._registry[name] = factory

    @classmethod
    def create(cls, name: str, robot_model, params: dict, **kwargs):
        """Create a cost function by name.

        Args:
            name (str): Name of the cost function to create.
            robot_model (MobileManipulator3D): Robot model.
            params (dict): Cost function parameters.
            **kwargs: Additional keyword arguments passed to factory.

        Returns:
            CostFunctions: Created cost function instance.

        Raises:
            ValueError: If cost function name is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown cost function: '{name}'. Available: {available}")
        factory = cls._registry[name]
        return factory(robot_model, params, **kwargs)

    @classmethod
    def list_available(cls):
        """List all available cost function names.

        Returns:
            list: List of registered cost function names.
        """
        return list(cls._registry.keys())


# Helper functions to create parameterized cost functions
def create_base_pose_cost(robot_mdl, params, dimension="SE2"):
    """Create base pose cost: dimension must be SE2 (x,y,yaw).

    Args:
        robot_mdl (MobileManipulator3D): Robot model.
        params (dict): Cost function parameters.
        dimension (str): Dimension type, must be "SE2".

    Returns:
        BasePoseSE2CostFunction: Base pose cost function instance.

    Raises:
        ValueError: If dimension is not "SE2".
    """
    if dimension != "SE2":
        raise ValueError(
            f"Base pose cost only supports SE2. Got dimension='{dimension}'. "
            "Use SE2 for base pose tracking with proper yaw handling."
        )
    return BasePoseSE2CostFunction(robot_mdl, params)


def create_base_vel_cost(robot_mdl, params, dimension=3):
    """Create base velocity cost: dimension can be 2 (vx,vy) or 3 (vx,vy,vyaw).

    Args:
        robot_mdl (MobileManipulator3D): Robot model.
        params (dict): Cost function parameters.
        dimension (int): Velocity dimension (2 or 3).

    Returns:
        VelocityCostFunction: Base velocity cost function instance.
    """
    dim_int = int(dimension)
    return VelocityCostFunction(robot_mdl, params, "base", dim_int, f"BaseVel{dim_int}")


def create_ee_pose_cost(robot_mdl, params, pose_type="SE3", frame="world"):
    """Create EE pose cost: pose_type must be SE3 (position+orientation), frame can be world or base.

    Args:
        robot_mdl (MobileManipulator3D): Robot model.
        params (dict): Cost function parameters.
        pose_type (str): Pose type, must be "SE3".
        frame (str): Reference frame, either "world" or "base".

    Returns:
        PoseSE3CostFunction: End-effector pose cost function instance.

    Raises:
        ValueError: If pose_type is not "SE3".
    """
    base_frame = frame == "base"
    if pose_type != "SE3":
        raise ValueError(
            f"EE pose cost only supports SE3. Got pose_type='{pose_type}'. "
            "Use SE3 for EE pose tracking (set orientation weights to 0 for position-only tracking)."
        )
    name = f"EEPoseSE3{'BaseFrame' if base_frame else ''}"
    return PoseSE3CostFunction(robot_mdl, params, base_frame, name)


def create_ee_vel_cost(robot_mdl, params):
    """Create EE velocity cost: 6D (3D linear + 3D angular) for SE3 compatibility.

    Args:
        robot_mdl (MobileManipulator3D): Robot model.
        params (dict): Cost function parameters.

    Returns:
        VelocityCostFunction: End-effector velocity cost function instance.
    """
    return VelocityCostFunction(robot_mdl, params, "EE", 6, "EEVel6")


def create_regularization_cost(robot_mdl, params):
    """Create regularization cost.

    Args:
        robot_mdl (MobileManipulator3D): Robot model.
        params (dict): Cost function parameters (not used, but kept for consistency).

    Returns:
        RegularizationCostFunction: Regularization cost function instance.
    """
    ss_mdl = robot_mdl.ssSymMdl
    return RegularizationCostFunction(ss_mdl["nx"], ss_mdl["nu"])


def create_manipulability_cost(robot_mdl, params):
    """Create manipulability cost.

    Args:
        robot_mdl (MobileManipulator3D): Robot model.
        params (dict): Cost function parameters (not used, but kept for consistency).

    Returns:
        ManipulabilityCostFunction: Manipulability cost function instance.
    """
    return ManipulabilityCostFunction(robot_mdl)


# Register core cost functions (simplified to 6 parameterized functions)
CostFunctionRegistry.register("BasePose", create_base_pose_cost)
CostFunctionRegistry.register("BaseVel", create_base_vel_cost)
CostFunctionRegistry.register("EEPose", create_ee_pose_cost)
CostFunctionRegistry.register("EEVel", create_ee_vel_cost)
CostFunctionRegistry.register("ControlEffort", ControlEffortCostFunction)
CostFunctionRegistry.register("Regularization", create_regularization_cost)
CostFunctionRegistry.register("Manipulability", create_manipulability_cost)
