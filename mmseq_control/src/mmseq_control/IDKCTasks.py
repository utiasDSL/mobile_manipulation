from abc import ABC, abstractmethod

import numpy as np
import casadi as cs

class IDKCTaskBase(ABC):
    """ Base class of Task specification for Inverse Differential Kinematic Control
        Reference:
        Escande, Adrien, Nicolas Mansard, and Pierre-Brice Wieber. “Hierarchical Quadratic Programming: Fast Online
        Humanoid-Robot Motion Generation.” The International Journal of Robotics Research 33, no. 7 (June 1, 2014): 1006–28.
        https://doi.org/10.1177/0278364914521306.
        Section 5.1


    """

    def __init__(self, nq, type, name):
        self.nq = nq
        self.type = type
        self.name = name

        self.q_sym = cs.MX.sym("q_sym", nq)
        self.qdot_sym = cs.MX.sym("qdot_sym", nq)
        super().__init__()

    @abstractmethod
    def linearize(self, q, *params):
        pass

    @abstractmethod
    def evalute(self, q, qdot, *params):
        pass

class EqTask(IDKCTaskBase):

    def __init__(self, e_fcn, nq, nΩ, κ, name="EqualityTask"):
        """ Equality task Eq.(64) in (Escande, 2014)

        :param e_fcn:
        :param nq:
        :param nΩ:
        :param κ:
        :param name:
        """
        self.e_fcn = e_fcn
        self.nΩ = nΩ
        self.κ = κ
        self.Ω_sym = cs.MX.sym("Ω_sym", nΩ)
        self.Ωdot_sym = cs.MX.sym("Ωdot_sym", nΩ)

        super().__init__(nq, "Eq", name)

        self.e_eqn = self.e_fcn(self.q_sym, self.Ω_sym)                     # error function to be driven to zero
        self.dedq_eqn = cs.jacobian(self.e_eqn, self.q_sym)                 # error function jacobian
        self.dedΩ_eqn = cs.jacobian(self.e_eqn, self.Ω_sym)
        self.ed_eqn = - self.κ * self.e_eqn - self.dedΩ_eqn @ self.Ωdot_sym # error function feedback signal (desired twist)

        self.J_fcn = cs.Function("J_"+ self.name, [self.q_sym, self.Ω_sym], [self.dedq_eqn])
        self.ed_fcn = cs.Function("ed_" + self.name, [self.q_sym, self.Ω_sym, self.Ωdot_sym], [self.ed_eqn])


    def linearize(self, q, *params):
        """ implement the linearized equality constraint Eq.(66) in (Escande, 2014)
                    Jq_dot = e^* - dedΩ * Ω_dot
        :param q: joint angle
        :param params: [Ω, Ωdot]
        :return: J, ed (RHS)
        """

        Ω, Ω_dot = params

        return self.J_fcn(q, Ω), self.ed_fcn(q, Ω, Ω_dot)

    def evalute(self, q, qdot, *params):
        """ compute error and linearized error values

        :param q:
        :param qdot:
        :param params:
        :return:
        """
        Ω, Ω_dot = params

        e = self.e_fcn(q, Ω)
        e_lin = self.J_fcn(q, Ω) @ qdot - self.ed_fcn(q, Ω, Ω_dot)

        return e, e_lin

class IneqTask(IDKCTaskBase):

    def __init__(self, e_fcn, ub, nq, κ, dt, name="InequalityTask"):
        """ Inequality task Eq.(67) in (Escande, 2014)
                        e(q) < ub
        :param e_fcn:
        :param ub:
        :param nq:
        :param name:
        """
        super().__init__(nq, "Ineq", name)

        self.e_fcn = e_fcn
        self.ub = ub
        self.κ = κ
        self.dt = dt

        self.e_eqn = e_fcn(self.q_sym)
        self.dedq_eqn = cs.jacobian(self.e_eqn, self.q_sym)
        self.ed_eqn = κ / dt * (ub - self.e_eqn)

        self.J_fcn = cs.Function("J_"+self.name, [self.q_sym], [self.dedq_eqn])
        self.ed_fcn = cs.Function("ed_"+self.name, [self.q_sym], [self.ed_eqn])

    def linearize(self, q, *params):
        """ Linearized inequality task Eq.(68) in (Escande, 2014)

        :param q:
        :param params:
        :return:
        """
        return self.J_fcn(q), self.ed_fcn(q)

    def evalute(self, q, qdot, *params):
        ineq_vio = np.linalg.norm(self.e_fcn(q) - self.ub)
        lin_ineq_vio = np.linalg.norm(self.J_fcn(q) @ qdot - self.ed_fcn(q))

        return ineq_vio, lin_ineq_vio

class PositionTrackingTask(EqTask):

    def __init__(self, fk, nq, nr, κ, name="PositionTracking"):
        r_sym = cs.MX.sym("r", nr)
        q_sym = cs.MX.sym('q', nq)
        e_eqn = fk(q_sym) - r_sym
        e_fcn = cs.Function('e_' + name, [q_sym, r_sym], [e_eqn])

        super().__init__(e_fcn, nq, nr, κ, name)

class BasePositionTracking(PositionTrackingTask):

    def __init__(self, robot, params):
        fb = robot.kinSymMdls[robot.base_link_name]
        κ = params["base_tracking"]["κ"]

        p_sym, _ = fb(robot.q_sym)

        fk = cs.Function("fk_base", [robot.q_sym], [p_sym])

        super().__init__(fk, robot.DoF, 2, κ, "BasePositionTracking")

class EEPositionTracking(PositionTrackingTask):

    def __init__(self, robot, params):
        fee = robot.kinSymMdls[robot.tool_link_name]
        κ = params["ee_tracking"]["κ"]

        p_sym, _ = fee(robot.q_sym)

        fk = cs.Function("fk_ee", [robot.q_sym], [p_sym])

        super().__init__(fk, robot.DoF, 3, κ, "EEPositionTracking")

class JointAngleBound(IneqTask):

    def __init__(self, robot, params):
        nq = robot.DoF
        ub_q = robot.ub_x[:nq]
        lb_q = robot.lb_x[:nq]

        e_fcn = cs.Function("e_fcn", [robot.q_sym], [cs.vertcat(robot.q_sym, -robot.q_sym)])
        ub = np.hstack((ub_q, -lb_q))

        super().__init__(e_fcn, ub, nq, params["joint_angle_bound_task"]["κ"], 1./params["ctrl_rate"], "JointAngleBound")

class JointVelocityBound(IDKCTaskBase):

    def __init__(self, robot, params):
        nq = robot.DoF
        super().__init__(nq, "Ineq", "JointVelocityBound")

        self.J = cs.vertcat(cs.DM.eye(nq), -cs.DM.eye(nq))

        self.ub = cs.vertcat(robot.ub_x[nq:], -robot.lb_x[nq:])

    def linearize(self, q, *params):
        return self.J, self.ub

    def evalute(self, q, qdot, *params):
        return self.J @ qdot - self.ub, self.J @ qdot - self.ub

class JointAccelerationBound(IDKCTaskBase):

    def __init__(self, robot, params):
        nq = robot.DoF
        super().__init__(nq, "Ineq", "JointAccelerationBound")
        self.dt = 1./ params["ctrl_rate"]
        self.J = cs.vertcat(cs.DM.eye(nq), -cs.DM.eye(nq))

        self.ub = cs.vertcat(robot.ub_u, -robot.lb_u) * self.dt

    def linearize(self, q, *params):
        qdot_prev = params[0]
        return self.J, self.ub + self.J @ qdot_prev

    def evalute(self, q, qdot, *params):
        vio = self.J @ (qdot - params[0]) - self.ub
        return vio, vio

def test_joint_acc_bound(config):
    print("-------------Testing Joint Velocity Bound Task---------------- ")
    robot = MobileManipulator3D(config["controller"])
    joint_acc_task = JointAccelerationBound(robot, config["controller"])
    nq = robot.DoF
    qdot_prev = np.random.randn(nq) * 3
    J, e = joint_acc_task.linearize([], qdot_prev)
    print(J.toarray(), e)

    qdot = np.random.randn(nq) * 3
    print(joint_acc_task.evalute([], qdot, qdot_prev))

def test_joint_vel_bound(config):
    print("-------------Testing Joint Velocity Bound Task---------------- ")
    robot = MobileManipulator3D(config["controller"])
    joint_vel_task = JointVelocityBound(robot, config["controller"])
    nq = robot.DoF
    J, e = joint_vel_task.linearize([], [])
    print(J.toarray(), e)

    qdot = np.random.randn(nq) * 3
    print(qdot)
    print(joint_vel_task.evalute([], qdot, []))

def test_joint_angle_bound(config):
    print("-------------Testing Joint Angle Bound Task---------------- ")
    robot = MobileManipulator3D(config["controller"])
    joint_angle_task = JointAngleBound(robot, config["controller"])

    # robot configuration
    nq = robot.DoF
    q = np.random.randn(nq)
    qdot = np.random.randn(nq)
    kappa = config["controller"]["joint_angle_bound_task"]["κ"]
    dt = 1./ config["controller"]["ctrl_rate"]

    J_sym, e_sym = joint_angle_task.linearize(q, config["controller"])
    J = np.vstack((np.eye(nq), -np.eye(nq)))
    eu = kappa / dt * (robot.ub_x[:nq] - q)
    el = kappa / dt * (robot.lb_x[:nq] - q)
    e = np.hstack((eu, -el))
    print("J diff: {}, e diff: {}".format(np.linalg.norm(J_sym - J), np.linalg.norm(e_sym - e)))

def test_position_tracking(config):
    print("-------------Testing Position Tracking Task---------------- ")

    robot = MobileManipulator3D(config["controller"])
    base_tracking_task = BasePositionTracking(robot, config["controller"])
    ee_tracking_task = EEPositionTracking(robot, config["controller"])
    # robot configuration
    q = np.random.randn(9)
    qdot = np.random.randn(9)

    # check base tracking task linearization and error function (linear and nonlinear) evaluation
    rb = np.random.randn(2)*2
    rb_dot = np.random.randn(2)
    Jbase_sym, ed_base_sym = base_tracking_task.linearize(q, *[rb, rb_dot])
    Jbase = np.hstack((np.eye(2), np.zeros((2, 7))))
    Pb, _ = robot.kinSymMdls[robot.base_link_name](q)
    ed_base = -config["controller"]["base_tracking"]["κ"] * (Pb - rb) + rb_dot
    print("Jb diff:{}, ed_b diff:{}".format(np.linalg.norm(Jbase - Jbase_sym), np.linalg.norm(ed_base - ed_base_sym)))
    eb_sym, eb_lin_sym = base_tracking_task.evalute(q, qdot, *[rb, rb_dot])
    eb = Pb - rb
    eb_lin = Jbase @ qdot - ed_base
    print("eb diff: {}, eb_lin diff: {}".format(np.linalg.norm(eb - eb_sym), np.linalg.norm(eb_lin - eb_lin_sym)))

    # check ee tracking task linearization and error function (linear and nonlinear) evaluation
    ree = np.random.randn(3)*2
    ree_dot = np.random.randn(3)
    Jee_sym, ed_ee_sym = ee_tracking_task.linearize(q, *[ree, ree_dot])
    Pee, _ = robot.kinSymMdls[robot.tool_link_name](q)
    Jee = robot.jacSymMdls[robot.tool_link_name](q)
    ed_ee = -config["controller"]["base_tracking"]["κ"] * (Pee - ree) + ree_dot
    print("Jee diff:{}, ed_ee diff:{}".format(np.linalg.norm(Jee - Jee_sym), np.linalg.norm(ed_ee - ed_ee_sym)))
    eee_sym, eee_lin_sym = ee_tracking_task.evalute(q, qdot, *[ree, ree_dot])
    eee = Pee - ree
    eee_lin = Jee @ qdot - ed_ee
    print("eee diff: {}, eee_lin diff: {}".format(np.linalg.norm(eee - eee_sym), np.linalg.norm(eee_lin - eee_lin_sym)))


if __name__ == "__main__":
    from mmseq_utils import parsing
    from mmseq_control.robot import MobileManipulator3D

    config = parsing.load_config("/home/tracy/Projects/mm_catkin_ws/src/mm_sequential_tasks/mmseq_run/config/simple_experiment.yaml")
    # test_position_tracking(config)
    # test_joint_angle_bound(config)
    # test_joint_vel_bound(config)
    test_joint_acc_bound(config)
