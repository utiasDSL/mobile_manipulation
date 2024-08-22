import argparse
import casadi as ca
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from mmseq_control.robot import CasadiModelInterface
from mmseq_utils import parsing
from mmseq_plan.CPCPlanner import CPCPlanner
from mmseq_plan.SequentialPlanner import SequentialPlanner
from mmseq_utils.plot_casadi_time_optimal import compare_trajectories_casadi_plot, plot_3d_trajectory, plot_whisker_plots, plot_obstacle_avoidance, plot_base_ee_velocities, simple_plot
from mmseq_utils.point_mass_computation_scripts.casadi_initial_guess import initial_guess, initial_guess_simple
from mmseq_control.mobile_manipulator_point_mass.mobile_manipulator_class import MobileManipulatorPointMass

def sample_3d_points(points_len, perimeter_size, z_max = 2):
    min_val = (-perimeter_size)*2 # double so points can be 0.5 apart after diving by two
    max_val = (perimeter_size)*2
    xs = np.random.randint(min_val, max_val, points_len)
    ys = np.random.randint(min_val, max_val, points_len)
    if z_max == 0:
        zs = np.zeros(points_len)
    else:
        zs = np.random.randint(0, 2*z_max, points_len)
    return np.array([xs/2, ys/2, zs/2]).T


def main():
    np.random.seed(0)

    # read in config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration file.")
    parser.add_argument("--optimization", required=False, help="Optimization type", default="cpc")
    parser.add_argument("--scenarios", required=False, help="Loop trajectory", default=10)
    parser.add_argument("--obstacle_avoid", action="store_true", help="Obstacle avoidance")
    args = parser.parse_args()

    # load configuration and overwrite with args
    config = parsing.load_config(args.config)
    controller_config = config["controller"]

    # initialize robot
    casadi_model_interface = CasadiModelInterface(controller_config)
    mobile_robot = MobileManipulatorPointMass(controller_config)

    # setup small example
    N = 200
    pi = 3.14
    starting_configuration = [0, 0, 0, 0.5*pi, -0.25*pi, 0.5*pi, -0.25*pi, 0.5*pi, 0.417*pi]
    qs_num = mobile_robot.DoF
    perimeter_size = 5
    small_perimeter_size = 1
    prediction_horizon = 5
    v_bounds = (mobile_robot.lb_x[mobile_robot.DoF:], mobile_robot.ub_x[mobile_robot.DoF:])
    a_bounds = (mobile_robot.lb_u, mobile_robot.ub_u)


    # motion model
    motion_model = mobile_robot.ssSymMdl["fmdl"]

    # forward kinematics
    forward_kinematic = mobile_robot.end_effector_pose_func()

    # generate obstacle avoidance
    obstacles_avoidance = casadi_model_interface.signedDistanceSymMdlsPerGroup["self"]
    if args.obstacle_avoid:
        obstacles_avoidance_for_opti = obstacles_avoidance
    else:
        obstacles_avoidance_for_opti = None

    # data collection
    timing_data = {}

    timing_data["pontryagin"] = {}
    timing_data["inverse_kinematic"] = {}

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


    for i in range(int(args.scenarios)):
        initial_guesses  = []
        final_solutions = []
        total_elements_list = []
        plotting_labels = []
        final_ts = []
        final_Ns = []
        dts = []
        print(f"_______________Scenario {i}________________")
        if i%2 == 0:
            label = 'ee_test'
        else:
            label = 'base_test'

        if label == 'ee_test':
            # generate points_len points in a 2m x 2m area
            points = sample_3d_points(prediction_horizon, small_perimeter_size)
            # sample starting point
            first_point = sample_3d_points(1, perimeter_size, z_max=0) #just so that the points are not always around the origin
            points += first_point
    
        elif label == 'base_test':
            # generate points_len points in a 10m x 10m spaced out between each other 
            points = sample_3d_points(prediction_horizon, perimeter_size)
            
        points_full = [mobile_robot.end_effector_pose(starting_configuration).full().flatten()]
        points_full.extend(points)


        ## Sampling algorithm
        start_time = time.time()    
        results, dict_res, shortest = mobile_robot.calculate_trajectory(points_full, starting_configuration, prediction_horizon, vertex_angle_deg=40, magnitude_step=3, angle_step=3, loop=False)
        end_time = time.time()-start_time
        print(f"Time taken to compute sampling: {end_time}")
        print()

        init_configs = ['pontryagin', 'inverse_kinematic']
        for j in init_configs:
            ## CPC optimization'
            if args.optimization == "cpc":
                cpc_plan = CPCPlanner(mobile_robot)
                if j=='inverse_kinematic':
                    start_time = time.time()
                    X0_array, tf = initial_guess_simple(mobile_robot, points_full, starting_configuration, prediction_horizon, N)
                    initial_guess_time = time.time()-start_time
                    print(f"Time taken to compute initial guess inverse kynematic for scenario {i}: {initial_guess_time}")
                    plotting_labels.append("Inverse Kinematic")

                elif j=='pontryagin':
                    start_time = time.time()
                    X0_array, tf = initial_guess(mobile_robot, points_full, starting_configuration, prediction_horizon, N)
                    # print(X0_array[100:150])
                    initial_guess_time = time.time()-start_time
                    print(f"Time taken to compute initial guess Pontryagin for scenario {i}: {initial_guess_time}")
                    plotting_labels.append("Pontryagin")


                X0 = ca.vertcat(tf, *X0_array)
                start_time = time.time()
                X, total_elements = cpc_plan.optimize(points_full, prediction_horizon, X0, np.inf, motion_model, forward_kinematic, N=N, obstacles_avoidance=obstacles_avoidance_for_opti, cpc_tolerance=0.0001)
                optimization_time = time.time()-start_time
                print(f"Time taken by CPC scenario {i}: {optimization_time}")
                print(f'End time scenario {i}: {X[0]}')
                ts = X[0]
                Ns = [N]
                dts_temp = [float(X[0]/N)]
                print()
                    

            ## Sequential optimization
            elif args.optimization == 'sequential':
                sequential_plan = SequentialPlanner(mobile_robot)

                if j=='inverse_kinematic':
                    start_time = time.time()
                    X0_array, ts_guess, Ns = initial_guess_simple(mobile_robot, points_full, starting_configuration, prediction_horizon, N, is_sequential_guess=True)
                    initial_guess_time = time.time()-start_time
                    print(f"Time taken to compute initial guess inverse kynematic for scenario {i}: {initial_guess_time}")
                    plotting_labels.append("Inverse Kinematic")

                elif j=='pontryagin':
                    start_time = time.time()
                    X0_array, ts_guess, Ns = initial_guess(mobile_robot, points_full, starting_configuration, prediction_horizon, N, is_sequential_guess=True)
                    initial_guess_time = time.time()-start_time
                    print(f"Time taken to compute initial guess Pontryagin for scenario {i}: {initial_guess_time}")
                    plotting_labels.append("Pontryagin")

                X0 = ca.vertcat(*ts_guess, *X0_array)
                start_time = time.time()
                X, total_elements, ts = sequential_plan.optimize_sequential(points_full, prediction_horizon, X0, motion_model, forward_kinematic, Ns, obstacles_avoidance=obstacles_avoidance_for_opti)
                optimization_time = time.time()-start_time
                print(f"Time taken from Sequential scenario {i}: {optimization_time}")
                print(f'End time scenario {i}: {X[0]}')
                print()
                X0 = ca.vertcat(sum(ts_guess), *X0_array)
                dts_temp = [float(ts[i]/Ns[i]) for i in range(len(Ns))]

            # data collection
            initial_guesses.append(X0)
            total_elements_list.append(total_elements)
            final_solutions.append(X)
            final_ts.append(ts)
            final_Ns.append(Ns)
            dts.append(dts_temp)


            timing_data[j][label]["initial_guess_ts"].append(initial_guess_time)
            timing_data[j][label]["optimization_ts"].append(optimization_time)
            timing_data[j][label]["dt std"].append(np.std(dts[-1]))
            timing_data[j][label]["final_ts"].append(float(X[0]))
            

    
        ## Plotting
        # plot the initial guess
        fig = plot_3d_trajectory(initial_guesses, points_full, None, None, forward_kinematic, total_elements_list, final_ts, final_Ns, q_size=qs_num, labels=plotting_labels, title=f"Initial Guess Scenario {i} {label}", show=False)
        fig.text(0.5, 0.1, f'inverse kinematic computation time: {timing_data["inverse_kinematic"][label]["initial_guess_ts"][-1]}, pontryagin computation time: {timing_data["pontryagin"][label]["initial_guess_ts"][-1]}', ha='center')
        fig.text(0.5, 0.01, f'Points: {points_full}', ha='center')
        pdf.savefig(fig)

        plt.close(fig)

        # plt.show()
        # # plot the final solution
        fig = plot_3d_trajectory(final_solutions, points_full, None, shortest, forward_kinematic, total_elements_list, final_ts, final_Ns, q_size=qs_num, labels=plotting_labels, title=f"Final Solution Scenario {i} {label}", show=False)
        fig.text(0.5, 0.1, f'inverse kinematic computation time: {timing_data["inverse_kinematic"][label]["optimization_ts"][-1]}, pontryagin computation time: {timing_data["pontryagin"][label]["optimization_ts"][-1]}', ha='center')
        pdf.savefig(fig)
        plt.close(fig)
        # plt.show()
        # plot velocities
        fig = compare_trajectories_casadi_plot(final_solutions, points_full, None, shortest, forward_kinematic, total_elements_list, final_ts, final_Ns, q_size=qs_num, labels=plotting_labels, show=False, v_bounds=v_bounds, a_bounds=a_bounds)
        tfs = [float(final_solutions[i][0]) for i in range(len(final_solutions))]
        string = "Final times ->"
        for k in range(len(tfs)):
            string += f"{init_configs[k]}: {tfs[k]} "
        fig.suptitle(string)
        pdf.savefig(fig)
        plt.close(fig)

        # plot end effector and base velocities
        fig = plot_base_ee_velocities(final_solutions, mobile_robot, total_elements_list, final_ts, final_Ns, q_size=qs_num, labels=plotting_labels, show=False)
        pdf.savefig(fig)
        plt.close(fig)

        # plot dts
        fig = plot_whisker_plots([dts], init_configs, show=False, titles=[f"dt Scenario {i} {label}"], ylabel="dt")
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
    fig = plot_whisker_plots([initialization_data, optimization_data], x_axis, show=False, titles=[f"Initialization {args.optimization}", f"Optimization {args.optimization}"], ylabel="Time [s]")
    pdf.savefig(fig)
    plt.close(fig)

    # Plot final times
    fig, ax = plt.subplots(1, 2)
    for i in range(len(scenario_labels)):
        final_times = []
        for key, value in timing_data.items():
            final_times.append(value[scenario_labels[i]]["final_ts"])
        fig = simple_plot(np.arange(len(final_times[0])), [final_times], show=False, y_label="Time [s]", x_label="Scenarios", labels=init_configs, titles=[f"Final Times {args.optimization} {scenario_labels[i]}"])
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
    fig = plot_whisker_plots([initialization_data, optimization_data], x_axis, show=False, titles=[f'Initialization {args.optimization}', f'Optimization {args.optimization}'], ylabel="Time [s]")
    pdf.savefig(fig)
    plt.close(fig)

    # plot dts
    fig = plot_whisker_plots([standard_deviation_dt], x_axis, show=False, titles=["Standard Deviation of dts"], ylabel="Standard Deviation")
    fig.text(0.5, 0.01, f'Scenarios violating obstacles: {scenarios_violating_obstacles}', ha='center')
    pdf.savefig(fig)
    plt.close(fig)

    # plot final times, two lines, one per initialization method
    x = np.arange(len(final_times[0]))    
    fig = simple_plot(x, [final_times], show=False, y_label="Time [s]", x_label="Scenarios", labels=init_configs, titles=[f"Final Times {args.optimization}"])
    pdf.savefig(fig)
    plt.close(fig)
    pdf.close()

    print()
    print(f"Scenarios violating obstacles: {scenarios_violating_obstacles}")


    
if __name__ == "__main__":
    main()