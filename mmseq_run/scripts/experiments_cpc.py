import argparse
import casadi as ca
import numpy as np
import time

from mmseq_control.robot import MobileManipulator3D
from mmseq_utils import parsing
from mmseq_plan.CPCPlanner import CPCPlanner
from mmseq_plan.SequentialPlanner import SequentialPlanner
from mmseq_utils.plot_casadi_time_optimal import compare_trajectories_casadi_plot
from mmseq_utils.point_mass_computation_scripts.casadi_initial_guess import initial_guess, initial_guess_simple
from mmseq_control.mobile_manipulator_point_mass.mobile_manipulator_class import MobileManipulatorPointMass


def main():
    # read in config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration file.")
    args = parser.parse_args()

    # load configuration and overwrite with args
    config = parsing.load_config(args.config)
    controller_config = config["controller"]

    # initialize robot
    mobile_robot = MobileManipulatorPointMass(controller_config)
    # print(mobile_robot.ub_x) #ub_x contains as the first 9 entries the limits on position and as the last 9 entries the limits on velocity
    # print(mobile_robot.ub_u) #ub_u contains the limits on the control inputs


    # setup small example
    points = [[5,5,1], [-5,-5, 0.5]]
    prediction_horizon = 2
    N = 200
    pi = 3.14
    starting_configuration = [0, 0, 0, 0.5*pi, -0.25*pi, 0.5*pi, -0.25*pi, 0.5*pi, 0.417*pi]
    qs_num = mobile_robot.DoF

    # motion model
    motion_model = mobile_robot.ssSymMdl["fmdl"]

    # forward kinematics
    forward_kinematic = mobile_robot.end_effector_pose_func()
    points_full = [mobile_robot.end_effector_pose(starting_configuration).full().flatten()]
    points_full.extend(points)

    ## Sampling algorithm
    start_time = time.time()    
    results, dict_res, shortest = mobile_robot.calculate_trajectory(points_full, starting_configuration, prediction_horizon, vertex_angle_deg=40, magnitude_step=3, angle_step=20, loop=False)
    print(f"Time taken to compute initial guess: {time.time()-start_time}")
    print()

    ## CPC proper initialization
    cpc_plan = CPCPlanner(mobile_robot)
    # X0_array, tf = initial_guess(mobile_robot, points_full, starting_configuration, prediction_horizon, N)
    X0_array, tf = initial_guess_simple(mobile_robot, points_full, starting_configuration, prediction_horizon, N)
    X0 = ca.vertcat(tf, *X0_array)
    start_time = time.time()
    X, total_elements = cpc_plan.optimize(points_full, prediction_horizon, X0, np.inf, motion_model, forward_kinematic, N=N)
    print(f"Time taken by CPC: {time.time()-start_time}")
    print()

    # ## Zero initializations
    # X0_array = np.zeros(len(X0_array))
    # X0 = ca.vertcat(tf, *X0_array)
    # start_time = time.time()
    # X_zero, total_elements = cpc_plan.optimize(points_full, prediction_horizon, X0, np.inf, motion_model, forward_kinematic, N=N)
    # print(f"Time taken: {time.time()-start_time}")

    ## Sequential optimization
    sequential_plan = SequentialPlanner(mobile_robot)
    # X0_array, ts, Ns = initial_guess(mobile_robot, points_full, starting_configuration, prediction_horizon, N, is_sequential_guess=True)
    X0_array, ts, Ns = initial_guess_simple(mobile_robot, points_full, starting_configuration, prediction_horizon, N, is_sequential_guess=True)
    X0 = ca.vertcat(*ts, *X0_array)
    start_time = time.time()
    X_seq, total_elements_sequential, _ = sequential_plan.optimize_sequential(points_full, prediction_horizon, X0, motion_model, forward_kinematic, Ns)
    print(f"Time taken from Sequential : {time.time()-start_time}")
    print()
    # sequential_plan.generate_trajectory(points, starting_configuration, prediction_horizon, N)
    # sequential_plan.process_results()
    # sequential_plan.generate_velocity(3)

    ## Sequential optimization with zero initialization
    # X0_array = np.zeros(len(X0_array))
    # X0 = ca.vertcat(*ts, *X0_array)
    # start_time = time.time()
    # X_seq_zero, total_elements_sequential, _ = sequential_plan.optimize_sequential(points_full, prediction_horizon, X0, motion_model, forward_kinematic, Ns)
    # print(f"Time taken from Sequential zero initial guess: {time.time()-start_time}")
    # print()
    # Plotting
    # compare_trajectories_casadi_plot([X_seq, X_seq_zero], points, dict_res, shortest, forward_kinematic, q_size=qs_num, state_dim=[total_elements_sequential, total_elements_sequential], labels=["Sequential", "Sequential zero init"], obstacles=[])
    # compare_trajectories_casadi_plot([X, X_zero], points, dict_res, shortest, forward_kinematic, q_size=qs_num, state_dim=[total_elements, total_elements], labels=["CPC", "CPC zero init"], obstacles=[])
    compare_trajectories_casadi_plot([X, X_seq], points, dict_res, shortest, forward_kinematic, q_size=qs_num, state_dim=[total_elements, total_elements_sequential], labels=["CPC", "Sequential"], obstacles=[])
    # compare_trajectories_casadi_plot([X_seq], points, dict_res, shortest, forward_kinematic, q_size=qs_num, state_dim=[total_elements_sequential], labels=["Sequential"], obstacles=[])

if __name__ == "__main__":
    main()