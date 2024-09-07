#   

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
    a_bounds = (mobile_robot.lb_u, mobile_robot.ub_u)

    # motion model
    motion_model = mobile_robot.ssSymMdl["fmdl"]

    # forward kinematics
    forward_kinematic = mobile_robot.end_effector_pose_func()

    # obstacles
    obstacles_avoidance = planner.motion_class.casadi_model_interface.signedDistanceSymMdlsPerGroup["self"]

    timing_data = {}
    init_configs = ['Pontryagin', 'InverseKinematics']

    timing_data[init_configs[0]] = {}
    timing_data[init_configs[1]] = {}

    scenarios_violating_obstacles = []
    scenario_labels = ['ee_test', 'base_test']

    for key, value in timing_data.items():
        for label in scenario_labels:
            value[label] = {}
            value[label]["initial_guess_ts"] = []
            value[label]["optimization_ts"] = []
            value[label]["dt std"] = []
            value[label]["final_ts"] = []
    
    pdf = PdfPages('results.pdf')
    seen =[]

    # loop trough files in folder
    for file in sorted(os.listdir(args.folder_input)):
        initial_guesses  = []
        final_solutions = []
        total_elements_list = []
        plotting_labels = []
        final_ts = []
        final_Ns = []
        dts = []
        tags = file.split('_')
        # get scenario number from file
        i = int(tags[1])
        label = tags[3]+"_"+tags[4]
        if i in seen:
            continue
        seen.append(i)
        print(f"_______________Scenario {i}________________")

        # loop for each file with same scenario number
        scenario_files = [file for file in os.listdir(args.folder_input) if f"scenario_{i}" in file]
        for file in scenario_files:
            tags = file.split('_')
            j = tags[2]
            plotting_labels.append(j)
            # get scenario data
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
            if optimization_type == 'cpc':
                ts = planner.X[0]
                dts_temp = [float(planner.X[0]/planner.N)]
                Ns = [planner.N]
            elif optimization_type == 'sequential':
                ts = planner.ts
                Ns = planner.Ns
                dts_temp =  [float(ts[i]/Ns[i]) for i in range(len(Ns))]
            
            # data collection
            initial_guesses.append(X0)
            total_elements_list.append(total_elements)
            final_solutions.append(X)
            final_ts.append(ts)
            final_Ns.append(Ns)
            dts.append(dts_temp)

            print(f"Time taken from {optimization_type} scenario {i}, initial guess {j}: {initial_guess_time}")
            print(f"Time taken from {optimization_type} scenario {i}: {optimization_time}")
            print(f'End time scenario {i}: {X[0]}')
            print()


            timing_data[j][label]["initial_guess_ts"].append(initial_guess_time)
            timing_data[j][label]["optimization_ts"].append(optimization_time)
            timing_data[j][label]["dt std"].append(np.std(dts[-1]))
            timing_data[j][label]["final_ts"].append(float(X[0]))
                

        ## Plotting
        # plot the initial guess
        fig = plot_3d_trajectory(initial_guesses, points_full, None, None, forward_kinematic, total_elements_list, final_ts, final_Ns, q_size=qs_num, labels=plotting_labels, title=f"Initial Guess Scenario {i} {label}", show=False)
        fig.text(0.5, 0.1, f'inverse kinematic computation time: {timing_data["InverseKinematics"][label]["initial_guess_ts"][-1]}, pontryagin computation time: {timing_data["Pontryagin"][label]["initial_guess_ts"][-1]}', ha='center')
        fig.text(0.5, 0.01, f'Points: {points_full}', ha='center')
        pdf.savefig(fig)

        plt.close(fig)

        # plt.show()
        # # plot the final solution
        fig = plot_3d_trajectory(final_solutions, points_full, None, None, forward_kinematic, total_elements_list, final_ts, final_Ns, q_size=qs_num, labels=plotting_labels, title=f"Final Solution Scenario {i} {label}", show=False)
        fig.text(0.5, 0.1, f'inverse kinematic computation time: {timing_data["InverseKinematics"][label]["optimization_ts"][-1]}, pontryagin computation time: {timing_data["Pontryagin"][label]["optimization_ts"][-1]}', ha='center')
        pdf.savefig(fig)
        plt.close(fig)
        # plt.show()
        # plot velocities
        fig = compare_trajectories_casadi_plot(final_solutions, points_full, None, None, forward_kinematic, total_elements_list, final_ts, final_Ns, q_size=qs_num, labels=plotting_labels, show=False, v_bounds=v_bounds, a_bounds=a_bounds)
        string = "Final times ->"
        for k in range(len(init_configs)):
            string += f"{init_configs[k]}: {timing_data[init_configs[k]][label]['final_ts'][-1]}, "
        fig.suptitle(string)
        pdf.savefig(fig)
        plt.close(fig)

        # plot end effector and base velocities
        fig = plot_base_ee_velocities(final_solutions, mobile_robot, total_elements_list, final_ts, final_Ns, q_size=qs_num, labels=plotting_labels, show=False)
        pdf.savefig(fig)
        plt.close(fig)

        # plot dts
        fig = plot_whisker_plots([dts], plotting_labels, show=False, titles=[f"dt Scenario {i} {label}"], ylabel="dt")
        pdf.savefig(fig)
        plt.close(fig)

        # plot obstacle avoidance
        fig, obstacles_violated = plot_obstacle_avoidance(final_solutions, obstacles_avoidance, total_elements_list, q_size=qs_num, labels=plotting_labels, show=False)
        pdf.savefig(fig)
        plt.close(fig)
        if obstacles_violated:
            scenarios_violating_obstacles.append(i)




    # Plot whisker plots for the time taken
    # figgure with separated plots
    # plot all initialization data in subcolumns
    initialization_data = []
    optimization_data = []
    x_axis = []

    for key, value in timing_data.items():
        time_per_init = []
        for label, data in value.items():
            x_axis.append(f'{key} {label}')
            initialization_data.append(data["initial_guess_ts"])
            optimization_data.append(data["optimization_ts"])
    fig = plot_whisker_plots([initialization_data, optimization_data], x_axis, show=False, titles=[f"Initialization {optimization_type}", f"Optimization {optimization_type}"], ylabel="Time [s]")
    pdf.savefig(fig)
    plt.close(fig)

    # Plot final times
    fig, ax = plt.subplots(1, 2)
    for i in range(len(scenario_labels)):
        final_times = []
        plotting_labels = []
        for key, value in timing_data.items():
            final_times.append(value[scenario_labels[i]]["final_ts"])
            plotting_labels.append(key)
        fig = simple_plot(np.arange(len(final_times[0])), [final_times], show=False, y_label="Time [s]", x_label="Scenarios", labels=plotting_labels, titles=[f"Final Times {optimization_type} {scenario_labels[i]}"])
        pdf.savefig(fig)
        plt.close(fig)
    # overall figgure
    fig, ax = plt.subplots(1, 2)
    initialization_data = []
    optimization_data = []
    standard_deviation_dt = []
    final_times = []
    x_axis = []

    for key, value in timing_data.items(): # key is the initialization method
        init =[]
        opt = []
        std_dts = []
        final_per_init = []
        x_axis.append(key)

        for label, data in value.items(): # label is the test type
            init.extend(data["initial_guess_ts"])
            opt.extend(data["optimization_ts"])
            std_dts.extend(data["dt std"])
            final_per_init.extend(data["final_ts"])
        initialization_data.append(init)
        optimization_data.append(opt)
        standard_deviation_dt.append(std_dts) #it should be a 2XNscenarios matrix
        final_times.append(final_per_init)
    fig = plot_whisker_plots([initialization_data, optimization_data], x_axis, show=False, titles=[f'Initialization {optimization_type}', f'Optimization {optimization_type}'], ylabel="Time [s]")
    pdf.savefig(fig)
    plt.close(fig)

    # plot dts
    fig = plot_whisker_plots([standard_deviation_dt], x_axis, show=False, titles=["Standard Deviation of dts"], ylabel="Standard Deviation")
    fig.text(0.5, 0.01, f'Scenarios violating obstacles: {scenarios_violating_obstacles}', ha='center')
    pdf.savefig(fig)
    plt.close(fig)

    # plot final times, two lines, one per initialization method
    x = np.arange(len(final_times[0]))    
    fig = simple_plot(x, [final_times], show=False, y_label="Time [s]", x_label="Scenarios", labels=x_axis, titles=[f"Final Times {optimization_type}"])
    pdf.savefig(fig)
    plt.close(fig)
    pdf.close()

    print()
    print(f"Scenarios violating obstacles: {scenarios_violating_obstacles}")

if __name__ == "__main__":
    main()
        


