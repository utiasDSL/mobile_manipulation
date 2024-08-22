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
    args = parser.parse_args()

    # load configuration and overwrite with args
    config = parsing.load_config(args.config)
    print(f"Configuration loaded from {args.config}")
    controller_config = config["controller"]

    # initialize robot
    print("Initializing robot...")
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
        if args.init == 'inverse':
            cpc_plan.generateTrajectory(points, starting_configuration, prediction_horizon, N, obstacles_avoidance=obstacles_avoidance)

        elif args.init == 'pont':
            cpc_plan.generateTrajectory(points, starting_configuration, prediction_horizon, N, obstacles_avoidance=obstacles_avoidance)

    ## Sequential optimization
    elif args.optimization == 'sequential':
        sequential_plan = SequentialPlanner(mobile_robot)

        if args.init == 'inverse':
            sequential_plan.generateTrajectory(points, starting_configuration, prediction_horizon, N, obs_avoidance=True, init_configs='inverse')

        elif args.init == 'pont':
            sequential_plan.generateTrajectory(points, starting_configuration, prediction_horizon, N, obs_avoidance=True, init_configs='pont')
        
        sequential_plan.processResults()
        sequential_plan.returnWaypointConfigurations()

if __name__ == "__main__":
    main()