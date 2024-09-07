import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from mmseq_utils import parsing
import mmseq_plan.CPCPlanner as CPCPlanner
import mmseq_plan.SequentialPlanner as SequentialPlanner
from mmseq_utils.plot_casadi_time_optimal import compare_trajectories_casadi_plot, plot_3d_trajectory, plot_whisker_plots, plot_obstacle_avoidance, plot_base_ee_velocities, simple_plot
from mmseq_utils.point_mass_computation_scripts.casadi_initial_guess import initial_guess, initial_guess_simple
from mmseq_control.mobile_manipulator_point_mass.mobile_manipulator_class import MobileManipulatorPointMass

def main():
    np.random.seed(0)

    # read in config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration file.")
    args = parser.parse_args()

    config = parsing.load_config(args.config)
    controller_config = config["controller"]
    planner_config = config["planner"]["tasks"][0]

    planner = getattr(CPCPlanner, planner_config["planner_type"], None)
    optimization_type = 'cpc'
    if planner is None:
        planner = getattr(SequentialPlanner, planner_config["planner_type"], None)
        optimization_type = 'sequential'
    if planner is None:
        raise ValueError("Planner type {} not found".format(planner_config["planner_type"]))
    # generate plan
    planner = planner(planner_config)

    mobile_robot = planner.motion_class
    qs_num = mobile_robot.DoF
    
    v_bounds = (mobile_robot.lb_x[mobile_robot.DoF:], mobile_robot.ub_x[mobile_robot.DoF:])

    # motion model
    motion_model = mobile_robot.ssSymMdl["fmdl"]

    # forward kinematics
    forward_kinematic = mobile_robot.end_effector_pose_func()
    base_xyz = mobile_robot.base_xyz_func()

    # obstacles
    obstacles_avoidance = planner.motion_class.casadi_model_interface.signedDistanceSymMdlsPerGroup["self"]
    ground_obs = planner.motion_class.casadi_model_interface.signedDistanceSymMdlsPerGroup["static_obstacles"]["ground"]

    init_configs = ['Pontryagin', 'InverseKinematics']

    experiment_data = {}

    scenarios_violating_obstacles = []

    # save pdf in proper folder
    path = planner.file_path
 
    pdf = PdfPages(f"{path}/{planner.file_name}_results.pdf")    

    print(f"_______________File: {planner.file_name}_________________")
    i = 0
    experiment_data['Scenario '+str(i)] = {}
    for j in init_configs:
        experiment_data['Scenario '+str(i)][j] = {}
        experiment_data['Scenario '+str(i)][j]["initial_guess_ts"] = []
        experiment_data['Scenario '+str(i)][j]["optimization_ts"] = []
        experiment_data['Scenario '+str(i)][j]["dt_std"] = []
        experiment_data['Scenario '+str(i)][j]["total_elements"] = []
        experiment_data['Scenario '+str(i)][j]["final_ts"] = []
        experiment_data['Scenario '+str(i)][j]["final_Ns"] = []
        experiment_data['Scenario '+str(i)][j]["final_solutions"] = []
        experiment_data['Scenario '+str(i)][j]["initial_guesses"] = []
        experiment_data['Scenario '+str(i)][j]["slow_downed"] = []
        experiment_data['Scenario '+str(i)][j]["ts_slowed"] = []
        


    
    # load data
    planner.loadSolution(planner.file_name)
    planner.processResults()
    plotting_labels = [planner.file_name]
    u_min = mobile_robot.lb_u*planner.base_scaling_factor
    u_max = mobile_robot.ub_u*planner.base_scaling_factor
    u_min[3:] = mobile_robot.lb_u[3:]*planner.ee_scaling_factor
    u_max[3:] = mobile_robot.ub_u[3:]*planner.ee_scaling_factor
    bounds = [(u_min, u_max)]


    j = planner.init_config

    # get data
    X0 = planner.X0
    X = planner.X
    total_elements = planner.total_elements
    initial_guess_time = planner.initialization_time
    optimization_time = planner.optimization_time
    points = planner.points
    starting_configuration = planner.starting_configuration
    points_full = [mobile_robot.end_effector_pose(starting_configuration).full().flatten()]
    points_full.extend(points)
    X_slow = planner.X_slow
    if optimization_type == 'cpc':
        ts = planner.X[0]
        dts_temp = [float(planner.X[0]/planner.N)]
        Ns = [planner.N]
    elif optimization_type == 'sequential':
        ts = planner.ts
        Ns = planner.Ns
        # dts_temp =  [float(ts[i]/Ns[i]) for i in range(len(Ns))]

    # store data
    experiment_data['Scenario '+str(i)][j]["initial_guess_ts"].append(initial_guess_time)
    experiment_data['Scenario '+str(i)][j]["optimization_ts"].append(optimization_time)
    # experiment_data['Scenario '+str(i)][j]["dt_std"].append(np.std(dts_temp))
    experiment_data['Scenario '+str(i)][j]["total_elements"].append(total_elements)
    experiment_data['Scenario '+str(i)][j]["final_ts"].append(ts)
    experiment_data['Scenario '+str(i)][j]["final_Ns"].append(Ns)
    experiment_data['Scenario '+str(i)][j]["final_solutions"].append(X)
    experiment_data['Scenario '+str(i)][j]["initial_guesses"].append(X0)
    if X_slow is not None:
        experiment_data['Scenario '+str(i)][j]["slow_downed"].append(planner.X_slow)
        experiment_data['Scenario '+str(i)][j]["ts_slowed"].append(planner.X_slow[0])

    # print quick recap
    print(f"Time taken from {optimization_type} file {planner.file_name}: {initial_guess_time}")
    print(f"Time taken from {optimization_type} file {planner.file_name}: {optimization_time}")
    print(f'End run time file {planner.file_name}: {X[0]}')
    print()

    # plot initial guess
    fig = plot_3d_trajectory(experiment_data['Scenario '+str(i)][j]["initial_guesses"], points_full, None, None, forward_kinematic, experiment_data['Scenario '+str(i)][j]["total_elements"],
                        experiment_data['Scenario '+str(i)][j]["final_ts"], experiment_data['Scenario '+str(i)][j]["final_Ns"], q_size=qs_num, labels=plotting_labels, show=False, title=f"Initial guess {j} scenario {i}", base_kin=base_xyz)
    fig.text(0.5, 0.01, f'Points: {points_full}', ha='center')
    pdf.savefig(fig)
    plt.close(fig)
    # plot final solution
    fig = plot_3d_trajectory(experiment_data['Scenario '+str(i)][j]["final_solutions"], points_full, None, None, forward_kinematic, experiment_data['Scenario '+str(i)][j]["total_elements"],
                        experiment_data['Scenario '+str(i)][j]["final_ts"], experiment_data['Scenario '+str(i)][j]["final_Ns"], q_size=qs_num, labels=plotting_labels, show=False, title=f"Final solution {j} scenario {i}", base_kin=base_xyz)
    pdf.savefig(fig)
    plt.close(fig)
    # plot velocities 
    fig = compare_trajectories_casadi_plot(experiment_data['Scenario '+str(i)][j]["final_solutions"], points_full, None, None, forward_kinematic, experiment_data['Scenario '+str(i)][j]["total_elements"], 
                                    experiment_data['Scenario '+str(i)][j]["final_ts"], experiment_data['Scenario '+str(i)][j]["final_Ns"], q_size=qs_num, v_bounds=v_bounds, a_bounds=bounds, labels=plotting_labels, show=False)
    tfs = [float(X[0]) for X in experiment_data['Scenario '+str(i)][j]["final_solutions"]]
    string = "Final time: "
    for k in range(len(tfs)):
        string += f" {tfs[k]:.3f}  "
    fig.suptitle(string)
    pdf.savefig(fig)
    plt.close(fig)
    # plot velocities for X slowed
    if X_slow is not None:
        X_slow_N = int((X_slow.shape[0]-1)/(qs_num*3))
        fig = compare_trajectories_casadi_plot(experiment_data['Scenario '+str(i)][j]["slow_downed"], points_full, None, None, forward_kinematic, [9*3]*len(experiment_data['Scenario '+str(i)][j]["slow_downed"]), 
                                        experiment_data['Scenario '+str(i)][j]["ts_slowed"], [X_slow_N]*len(experiment_data['Scenario '+str(i)][j]["slow_downed"]), q_size=qs_num, v_bounds=v_bounds, a_bounds=bounds, labels=plotting_labels, show=False)
        fig.suptitle("Velocities for slow downed trajectories")
        pdf.savefig(fig)
        plt.close(fig)
    # plot obstacle avoidance
    fig, obstacles_violated = plot_obstacle_avoidance(experiment_data['Scenario '+str(i)][j]["final_solutions"], obstacles_avoidance, experiment_data['Scenario '+str(i)][j]["total_elements"], show=False, labels=plotting_labels, q_size=qs_num, limit=planner.self_collision_safety_margin)
    fig.suptitle("Obstacle avoidance for optimized trajectories")
    pdf.savefig(fig)
    plt.close(fig)
    if obstacles_violated:
        print(f"Obstacles violated in scenario {i} with {j}")
        scenarios_violating_obstacles.append(i)
    # plot distance from ground
    fig, objects_violated = plot_obstacle_avoidance(experiment_data['Scenario '+str(i)][j]["final_solutions"], ground_obs, experiment_data['Scenario '+str(i)][j]["total_elements"], show=False, labels=plotting_labels, q_size=qs_num)
    fig.suptitle("Distance from ground for optimized trajectories")
    pdf.savefig(fig)
    plt.close(fig)
    # plot slow down
    if X_slow is not None:
        fig, obstacles_violated = plot_obstacle_avoidance(experiment_data['Scenario '+str(i)][j]["slow_downed"], obstacles_avoidance, [9*3]*len(experiment_data['Scenario '+str(i)][j]["slow_downed"]), show=False, labels=plotting_labels, q_size=qs_num, limit=planner.self_collision_safety_margin)
        fig.suptitle("Obstacle avoidance for slow downed trajectories")
        if obstacles_violated:
            print(f"Self collision in slow down violed in scenario {i} with {j}")
            scenarios_violating_obstacles.append(i)
        pdf.savefig(fig)
        plt.close(fig)
    # plot initialization times as sf increases
    # fig = simple_plot(scaling_factors, [[experiment_data['Scenario '+str(i)][j]["initial_guess_ts"]]], y_label="Time", x_label="Scaling Factor", titles=[f'Initialization time for Scenario {i}'], show=False)
    # pdf.savefig(fig)
    # plt.close(fig)
    # # plot optimization times as sf increases
    # fig = simple_plot(scaling_factors, [[experiment_data['Scenario '+str(i)][j]["optimization_ts"]]], y_label="Time", x_label="Scaling Factor", titles=[f'Optimization time for Scenario {i}'], show=False)
    # pdf.savefig(fig)
    # plt.close(fig)
    # # plot final times as sf increases
    # fig = simple_plot(scaling_factors, [[[float(X[0]) for X in experiment_data['Scenario '+str(i)][j]["final_solutions"]]]], y_label="Time", x_label="Scaling Factor", show=False, titles=[f"Final time for Scenario {i}"])
    # pdf.savefig(fig)
    # plt.close(fig)

    pdf.close()

if __name__ == "__main__":
    main()






        
