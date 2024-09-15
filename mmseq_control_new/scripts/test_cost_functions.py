import numpy as np
from mmseq_control.robot import MobileManipulator3D
from mmseq_control_new.MPCCostFunctions import *
from mmseq_utils.casadi_struct import casadi_sym_struct
import casadi as cs

def get_default_xu():
    q = [0.0, 0.0, np.pi-0.01] + [0.0, -2.3562, -1.5708, -2.3562, -1.5708, 1.5708]
    v = np.ones(9)
    x = np.hstack((np.array(q), v))
    u = np.zeros(9)
    return x,u

def test_BasePos2(config):
    robot = MobileManipulator3D(config["controller"])
    cost_base = BasePos2CostFunction(robot, config["controller"]["cost_params"]["BasePos2"])
    x,u = get_default_xu()

    p_map_base = cost_base.p_struct(0)
    p_map_base['W'] = config["controller"]["cost_params"]["BasePos2"]["Qk"] * np.eye(cost_base.nr)
    p_map_base['r'] = np.array([1,1])
    J_base = cost_base.evaluate(x, u, p_map_base.cat)
    print("testing BasePos2")
    print("expected cost: {} evaluated cost: {}".format(1, J_base))

    return cost_base

def test_BaseVel2(config):
    robot = MobileManipulator3D(config["controller"])
    cost_base = BaseVel2CostFunction(robot, config["controller"]["cost_params"]["BaseVel2"])
    x,u = get_default_xu()

    p_map_base = cost_base.p_struct(0)
    p_map_base['W'] = config["controller"]["cost_params"]["BaseVel2"]["Qk"] * np.eye(cost_base.nr)
    p_map_base['r'] = np.array([2,2])
    J_base = cost_base.evaluate(x, u, p_map_base.cat)
    print("testing BaseVel2")
    print("expected cost: {} evaluated cost: {}".format(1, J_base))
    return cost_base

def test_BasePos3(config):
    robot = MobileManipulator3D(config["controller"])
    cost_base = BasePos3CostFunction(robot, config["controller"]["cost_params"]["BasePos3"])
    x,u = get_default_xu()

    p_map_base = cost_base.p_struct(0)
    p_map_base['W'] = config["controller"]["cost_params"]["BasePos3"]["Qk"] * np.eye(cost_base.nr)
    p_map_base['r'] = np.array([1, 1, 1])
    J_base = cost_base.evaluate(x, u, p_map_base.cat)
    print("testing BasePos3")
    print("expected cost: {} evaluated cost: {}".format(1.5, J_base))

    return cost_base

def test_BasePoseSE2(config):
    robot = MobileManipulator3D(config["controller"])
    cost_base = BasePoseSE2CostFunction(robot, config["controller"]["cost_params"]["BasePos3"])
    x,u = get_default_xu()

    p_map_base = cost_base.p_struct(0)
    p_map_base['W'] = config["controller"]["cost_params"]["BasePos3"]["Qk"] * np.eye(cost_base.nr)
    p_map_base['r'] = np.array([1, 1, -np.pi+-0.01])
    J_base = cost_base.evaluate(x, u, p_map_base.cat)
    print("testing BasePoseSE2")
    print("expected cost: {} evaluated cost: {}".format(1.0, J_base))

    return cost_base

def test_BaseVel3(config):
    robot = MobileManipulator3D(config["controller"])
    cost_base = BaseVel3CostFunction(robot, config["controller"]["cost_params"]["BaseVel3"])
    x,u = get_default_xu()

    p_map_base = cost_base.p_struct(0)
    p_map_base['W'] = config["controller"]["cost_params"]["BaseVel3"]["Qk"] * np.eye(cost_base.nr)
    p_map_base['r'] = np.array([1, 0, 1])
    J_base = cost_base.evaluate(x, u, p_map_base.cat)
    print("testing BaseVel3")
    print("expected cost: {} evaluated cost: {}".format(0.5, J_base))

    return cost_base

def test_EEPos3(config):
    robot = MobileManipulator3D(config["controller"])
    cost_ee = EEPos3CostFunction(robot, config["controller"]["cost_params"]["EEPos3"])
    p_map_ee = cost_ee.p_struct(0)
    p_map_ee['W'] = config["controller"]["cost_params"]["EEPos3"]["Qk"] * np.eye(cost_ee.nr)
    p_map_ee['r'] = np.array([1.194, 0.374, 1.596])

    x,u = get_default_xu()
    J_ee = cost_ee.evaluate(x, u, p_map_ee.cat)
    print("testing EEPos3")
    print("expected cost: {} evaluated cost: {}".format(0, J_ee))

    return cost_ee

def test_EEVel3(config):
    robot = MobileManipulator3D(config["controller"])
    cost_ee = EEVel3CostFunction(robot, config["controller"]["cost_params"]["EEVel3"])
    p_map_ee = cost_ee.p_struct(0)
    p_map_ee['W'] = config["controller"]["cost_params"]["EEVel3"]["Qk"] * np.eye(cost_ee.nr)
    p_map_ee['r'] = np.array([1.194, 0.374, 1.596])

    x,u = get_default_xu()
    J_ee = cost_ee.evaluate(x, u, p_map_ee.cat)
    print(J_ee)

    return cost_ee

def test_ControlEffort(config):
    robot = MobileManipulator3D(config["controller"])
    cost_eff = ControlEffortCostFunction(robot, config["controller"]["cost_params"]["Effort"])
    x,u = get_default_xu()

    J_eff = cost_eff.evaluate(x, u, [])
    print(J_eff)

    return cost_eff

def test_HessApprox(config):
    cost_fcn = test_BasePos2(config)
    H_eqn, _ = cs.hessian(cost_fcn.J_eqn, cs.veccat(cost_fcn.u_sym, cost_fcn.x_sym))
    H_fcn = cs.Function('H', [cost_fcn.x_sym, cost_fcn.u_sym, cost_fcn.p_sym], [H_eqn])
    x,u = get_default_xu()

    H_val = H_fcn(x, u, cost_fcn.p_struct(0)).toarray()
    H_approx_val = cost_fcn.H_approx_fcn(x, u, cost_fcn.p_struct(0)).toarray()
    # print(f"Hess Eval{H_val}")
    # print(f"Hess Approx Eval{H_approx_val}")
    print(f"Diff{np.linalg.norm(H_val - H_approx_val)}")

if __name__ == "__main__":
    dt = 0.1
    N = 10
    # robot mdl
    from mmseq_utils import parsing

    config = parsing.load_config(
        "$(rospack find mmseq_control_new)/scripts/test_mpc.yaml")

    test_BasePos2(config)
    test_BasePos3(config)
    test_BasePoseSE2(config)
    test_BaseVel2(config)
    test_BaseVel3(config)
    test_EEPos3(config)
    test_EEVel3(config)
    test_HessApprox(config)