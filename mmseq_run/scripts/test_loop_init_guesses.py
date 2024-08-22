import argparse
import casadi as ca
import numpy as np
import time

from mmseq_control.robot import CasadiModelInterface
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
    parser.add_argument("--optimization", required=False, help="Optimization type", default="cpc")
    parser.add_argument("--init", required=False, help="Initial guess type", default="pont")
    parser.add_argument("--loop", required=False, help="Loop trajectory", default=10)
    args = parser.parse_args()

    # load configuration and overwrite with args
    config = parsing.load_config(args.config)
    controller_config = config["controller"]

    # initialize robot
    casadi_model_interface = CasadiModelInterface(controller_config)
    mobile_robot = MobileManipulatorPointMass(controller_config)

    # setup small example
    points = [[5,5,1], [-5,-5, 0.5], [7, 3, 1.5], [0,0,1], [10, 10, 0.8], [1,2, 1]]
    prediction_horizon = len(points)
    N = 400
    pi = 3.14
    starting_configuration = [0, 0, 0, 0.5*pi, -0.25*pi, 0.5*pi, -0.25*pi, 0.5*pi, 0.417*pi]
    qs_num = mobile_robot.DoF

    # motion model
    motion_model = mobile_robot.ssSymMdl["fmdl"]

    # forward kinematics
    forward_kinematic = mobile_robot.end_effector_pose_func()

    # generate obstacle avoidance
    obstacles_avoidance = casadi_model_interface.signedDistanceSymMdlsPerGroup["self"]


    points_full = [mobile_robot.end_effector_pose(starting_configuration).full().flatten()]
    points_full.extend(points)

    ## Sampling algorithm
    start_time = time.time()    
    results, dict_res, shortest = mobile_robot.calculate_trajectory(points_full, starting_configuration, prediction_horizon, vertex_angle_deg=40, magnitude_step=3, angle_step=3, loop=False)
    print(f"Time taken to compute sampling: {time.time()-start_time}")
    print()

    ## CPC optimization'
    if args.optimization == "cpc":
        cpc_plan = CPCPlanner(mobile_robot)
        t = []
        for i in range(int(args.loop)):
            if args.init == 'inverse':
                start_time = time.time()
                X0_array, tf = initial_guess_simple(mobile_robot, points_full, starting_configuration, prediction_horizon, N)
                print(f"Time taken to compute initial guess inverse kynematic: {time.time()-start_time}")

            elif args.init == 'pont':
                start_time = time.time()
                X0_array, tf = initial_guess(mobile_robot, points_full, starting_configuration, prediction_horizon, N)
                # print(X0_array[100:150])
                print(f"Time taken to compute initial guess Pontryagin: {time.time()-start_time}")

        
            X0 = ca.vertcat(tf, *X0_array)
            start_time = time.time()
            X, total_elements = cpc_plan.optimize(points_full, prediction_horizon, X0, np.inf, motion_model, forward_kinematic, N=N)
            t.append(time.time()-start_time)
            print(f"Time taken by CPC iteration {i}: {t[-1]}")
            print(f'End time: {X[0]}')
            print()

    ## Sequential optimization
    elif args.optimization == 'sequential':
        sequential_plan = SequentialPlanner(mobile_robot)

        t = []
        for i in range(int(args.loop)):
            if args.init == 'inverse':
                start_time = time.time()
                X0_array, ts, Ns = initial_guess_simple(mobile_robot, points_full, starting_configuration, prediction_horizon, N, is_sequential_guess=True)
                print(f"Time taken to compute initial guess inverse kynematic: {time.time()-start_time}")

            elif args.init == 'pont':
                start_time = time.time()
                X0_array, ts, Ns = initial_guess(mobile_robot, points_full, starting_configuration, prediction_horizon, N, is_sequential_guess=True)
                print(f"Time taken to compute initial guess Pontryagin: {time.time()-start_time}")


            X0 = ca.vertcat(*ts, *X0_array)
            start_time = time.time()
            X, total_elements, _ = sequential_plan.optimize_sequential(points_full, prediction_horizon, X0, motion_model, forward_kinematic, Ns)
            t.append(time.time()-start_time)
            print(f"Time taken from Sequential iteration {i}: {t[-1]}")
            print(f'End time: {X[0]}')
            print()


    
    ## Plotting

    # plot time results
    x = np.arange(int(args.loop))
    y = np.array(t)
    print(f"Average time taken by optimization: {np.mean(y)}")
    print(f"Standard deviation of time taken by optimization: {np.std(y)}")
    import matplotlib.pyplot as plt
    plt.plot(x, y, 'o-')
    plt.xlabel("Iteration")
    plt.ylabel("Time (s)")
    plt.title("Time taken by optimization")
    plt.show()

    # plot distribution of ys
    n = len(y)
    bin_width = 3.5 * np.std(y) / np.cbrt(n) # scotts rule
    bins = int((max(y) - min(y)) / bin_width)
    plt.hist(y, bins=bins)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.title("Distribution of time taken by optimization")
    plt.show()
    compare_trajectories_casadi_plot([X], points, dict_res, shortest, forward_kinematic, q_size=qs_num, state_dim=[total_elements], labels=["X"], obstacles=[])
    # compare_trajectories_casadi_plot([X, X0], points, dict_res, shortest, forward_kinematic, q_size=qs_num, state_dim=[total_elements, total_elements], labels=["X", "Initial Guess"], obstacles=[])
    
if __name__ == "__main__":
    main()