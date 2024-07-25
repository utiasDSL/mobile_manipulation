import argparse
import casadi as ca
import numpy as np
import time

from mmseq_control.robot import MobileManipulator3D
from mmseq_utils import parsing
from mmseq_plan.CPCPlanner import optimize
from mmseq_utils.plot_casadi_time_optimal import compare_trajectories_casadi_plot


def main():
    # read in config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration file.")
    args = parser.parse_args()

    # load configuration and overwrite with args
    config = parsing.load_config(args.config)
    controller_config = config["controller"]

    # initialize robot
    mobile_robot = MobileManipulator3D(controller_config)
    # print(mobile_robot.ub_x) #ub_x contains as the first 9 entries the limits on position and as the last 9 entries the limits on velocity
    # print(mobile_robot.ub_u) #ub_u contains the limits on the control inputs


    # setup small example
    points = [[0.816194, -1.1748, 0.681293], [5, 5, 0.7], [2, 2, 0.7]]
    prediction_horizon = 2
    N = 100

    # motion model
    inputs = ca.MX.sym("inputs", 2*mobile_robot.DoF)
    u = ca.MX.sym("u", mobile_robot.DoF)
    qs_num = mobile_robot.DoF
    n_zeros = ca.DM.zeros(qs_num, qs_num)
    n_ones = ca.DM.eye(qs_num)
    matrix = ca.vertcat(
        ca.horzcat(n_zeros, n_ones),
        ca.DM.zeros(qs_num, 2*qs_num)
    )
    rhs_motion = matrix @ inputs + ca.vertcat(ca.DM.zeros(qs_num), u)
    motion_model = ca.Function("motion_model", [inputs, u], [rhs_motion])
    # forward kinematics
    forward_kinematic = mobile_robot.kinSymMdls['gripped_object']
    print(forward_kinematic([0, 0, 0, 0, 0, 0, 0, 0, 0])[0])

    # define X0 with all zeros
    total_elements = 3*qs_num + (3)*(prediction_horizon)
    X0 = ca.DM.zeros((total_elements*N) + 1)
    start_time = time.time()
    X, total_elements = optimize(mobile_robot, points, prediction_horizon, X0, np.inf, motion_model, forward_kinematic, N=N)
    print(f"Time taken: {time.time()-start_time}")
    # plotting
    print("X shape: ", X.shape)
    compare_trajectories_casadi_plot([X], points, None, None, forward_kinematic, q_size=qs_num, state_dim=[total_elements], labels=["CPC"], obstacles=[])



if __name__ == "__main__":
    main()