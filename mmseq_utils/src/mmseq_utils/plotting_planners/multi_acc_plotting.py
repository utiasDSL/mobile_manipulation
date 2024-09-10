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
    parser.add_argument("--folder_input", required=False, help="Folder to save input data")
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
    # override the planner path
    if args.folder_input:
        planner.file_path = args.folder_input

    mobile_robot = planner.motion_class
    qs_num = mobile_robot.DoF
    
    v_bounds = (mobile_robot.lb_x[mobile_robot.DoF:], mobile_robot.ub_x[mobile_robot.DoF:])

    # motion model
    motion_model = mobile_robot.ssSymMdl["fmdl"]

    # forward kinematics
    forward_kinematic = mobile_robot.end_effector_pose_func()

    # obstacles
    obstacles_avoidance = planner.motion_class.casadi_model_interface.signedDistanceSymMdlsPerGroup["self"]
    ground_obs = planner.motion_class.casadi_model_interface.signedDistanceSymMdlsPerGroup["static_obstacles"]["ground"]

    init_configs = ['Pontryagin', 'InverseKinematics']

    experiment_data = {}

    scenarios_violating_obstacles = []
    seen = []

    # save pdf in proper folder
    path = planner.file_path
    if not os.path.exists(path):
        os.makedirs(path)
    pdf = PdfPages(f"{path}/{optimization_type}_loaded_results.pdf")    

    # loop trough files in folder
    for file in sorted([file for file in os.listdir(args.folder_input) if f"scenario" in file]):
        tags = file.split('_')
        # get scenario number from file
        i = int(tags[1])
        if i in seen:
            continue
        seen.append(i)
        print(f"_______________Scenario {i}________________")
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
            
        scenario_files = sorted([file for file in os.listdir(args.folder_input) if f"scenario_{i}" in file])

        scaling_factors = []
        plotting_labels = []
        bounds = []

        for file in scenario_files: # assume that files will come ordered by the scaling factor
            tags = file.split('_')
            j = tags[2]
            scaling_factor = float(tags[4])

            if scaling_factor not in scaling_factors:
                scaling_factors.append(scaling_factor)
                plotting_labels.append(f"sf: {scaling_factor}")
                u_min = mobile_robot.lb_u*scaling_factor
                u_max = mobile_robot.ub_u*scaling_factor
                u_min[3:] = mobile_robot.lb_u[3:]*0.5
                u_max[3:] = mobile_robot.ub_u[3:]*0.5
                bounds.append((u_min, u_max))
            
            # load data
            planner.loadSolution(file)
            planner.processResults()

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
            experiment_data['Scenario '+str(i)][j]["slow_downed"].append(planner.X_slow)
            experiment_data['Scenario '+str(i)][j]["ts_slowed"].append(planner.X_slow[0])

            # print quick recap
            print(f"Time taken from {optimization_type} scenario {i}, initial guess {j}, sf {scaling_factor}: {initial_guess_time}")
            print(f"Time taken from {optimization_type} scenario {i}, initial guess {j}, sf {scaling_factor}: {optimization_time}")
            print(f'End time scenario {i}, initial guess {j}, sf {scaling_factor}: {X[0]}')
            print()

        ## Plotting data for single scenario
        for j in init_configs:
            # plot initial guess
            fig = plot_3d_trajectory(experiment_data['Scenario '+str(i)][j]["initial_guesses"], points_full, None, None, forward_kinematic, experiment_data['Scenario '+str(i)][j]["total_elements"],
                                experiment_data['Scenario '+str(i)][j]["final_ts"], experiment_data['Scenario '+str(i)][j]["final_Ns"], q_size=qs_num, labels=plotting_labels, show=False, title=f"Initial guess {j} scenario {i}")
            fig.text(0.5, 0.01, f'Points: {points_full}', ha='center')
            pdf.savefig(fig)
            plt.close(fig)
            # plot final solution
            fig = plot_3d_trajectory(experiment_data['Scenario '+str(i)][j]["final_solutions"], points_full, None, None, forward_kinematic, experiment_data['Scenario '+str(i)][j]["total_elements"],
                                experiment_data['Scenario '+str(i)][j]["final_ts"], experiment_data['Scenario '+str(i)][j]["final_Ns"], q_size=qs_num, labels=plotting_labels, show=False, title=f"Final solution {j} scenario {i}")
            pdf.savefig(fig)
            plt.close(fig)
            # plot velocities 
            fig = compare_trajectories_casadi_plot(experiment_data['Scenario '+str(i)][j]["final_solutions"], points_full, None, None, forward_kinematic, experiment_data['Scenario '+str(i)][j]["total_elements"], 
                                            experiment_data['Scenario '+str(i)][j]["final_ts"], experiment_data['Scenario '+str(i)][j]["final_Ns"], q_size=qs_num, v_bounds=v_bounds, a_bounds=bounds, labels=plotting_labels, show=False)
            tfs = [float(X[0]) for X in experiment_data['Scenario '+str(i)][j]["final_solutions"]]
            string = "Final times ->"
            for k in range(len(tfs)):
                string += f"{scaling_factors[k]}: {tfs[k]:.3f}  "
            fig.suptitle(string)
            pdf.savefig(fig)
            plt.close(fig)
            # plot velocities for X slowed
            fig = compare_trajectories_casadi_plot(experiment_data['Scenario '+str(i)][j]["slow_downed"], points_full, None, None, forward_kinematic, [9*3]*len(experiment_data['Scenario '+str(i)][j]["slow_downed"]), 
                                            experiment_data['Scenario '+str(i)][j]["ts_slowed"], [100]*len(experiment_data['Scenario '+str(i)][j]["slow_downed"]), q_size=qs_num, v_bounds=v_bounds, a_bounds=bounds, labels=plotting_labels, show=False)
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

    # overall plots
    # Initial guess time of all pontryagin scnarions
    init_data = []
    opti_data = []
    final_data = []
    for init_method in init_configs:
        init_times = []
        opti_times = []
        labels = []
        final_times = []
        for i in seen:
            init_times.append(experiment_data['Scenario '+str(i)][init_method]["initial_guess_ts"])
            opti_times.append(experiment_data['Scenario '+str(i)][init_method]["optimization_ts"])
            labels.append(f"Scenario {i}")
            final_times.append([float(X[0]) for X in experiment_data['Scenario '+str(i)][init_method]["final_solutions"]])
        init_data.append(init_times)
        opti_data.append(opti_times)
        final_data.append(final_times)
    # fig = simple_plot(scaling_factors, init_data, x_label="N", y_label="Time", show=False, labels=labels, titles=[f"Initial guess time Pontryagin {optimization_type}", f"Initial guess time Inverse kinematic {optimization_type}"])
    # # set title of the plot
    # fig.suptitle("Overall graphs")
    # pdf.savefig(fig)
    # plt.close(fig)
    # fig = simple_plot(scaling_factors, opti_data, x_label="N", y_label="Time", show=False, labels=labels, titles=[f"Optimization time Pontryagin {optimization_type}", f"Optimization time Inverse kinematic {optimization_type}"])
    # pdf.savefig(fig)
    # plt.close(fig)
    # # plot all final times
    # fig = simple_plot(scaling_factors, final_data, x_label="N", y_label="Time", show=False, labels=labels, titles=[f"Final time Pontryagin {optimization_type}", f"Final time Inverse kinematic {optimization_type}"])
    # pdf.savefig(fig)
    # plt.close(fig)
    # whisker plots of dts
    # data = [] # will contain 2 lists, one for pontryagin and one for inverse kinematic
    # for init_method in init_configs:
    #     dt_std = [] # is going to be a 4x7 array, but we want 7x4
    #     for i in range(len(seen)):
    #         dt_std.append(experiment_data['Scenario '+str(i)][init_method]["dt_std"]) # will contain a list of stds (as many as the number of Ns) per scenario
    #     dt_std = np.array(dt_std).T.tolist()
    #     data.append(dt_std)
    # fig = plot_whisker_plots(data, plotting_labels, show=False, titles=["Standard dev of dts with Pontryagin", "Standard dev of dts with Inverse kinematic"], ylabel="Standard deviation of dts")
    # pdf.savefig(fig)
    # plt.close(fig)


    # close pdf
    pdf.close()

if __name__ == "__main__":
    main()






        
