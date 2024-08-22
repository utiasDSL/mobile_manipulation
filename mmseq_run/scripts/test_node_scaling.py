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
    parser.add_argument("--scenarios", required=False, help="Loop trajectory", default=1)
    parser.add_argument("--max_N", required=False, help="Max N", default=100)
    parser.add_argument("--obstacle_avoid", action="store_true", help="Obstacle avoidance")
    args = parser.parse_args()

    # load configuration and overwrite with args
    config = parsing.load_config(args.config)
    controller_config = config["controller"]

    # initialize robot
    casadi_model_interface = CasadiModelInterface(controller_config)
    mobile_robot = MobileManipulatorPointMass(controller_config)

    # setup small example
    list_of_Ns = np.linspace(100, int(args.max_N), int(int(args.max_N)/100), dtype=int)
    print(list_of_Ns)
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

    experiment_data ={}

    scenarios_violating_obstacles = []
    
    pdf = PdfPages('results.pdf')


    for i in range(int(args.scenarios)):
        plotting_labels = []
        init_configs = ['pontryagin', 'inverse_kinematic']

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

        plotting_labels = []
        for N in list_of_Ns:
            print(f"___________N {N}____________")
            plotting_labels.append(f"N: {N}")
            ## Sampling algorithm
            start_time = time.time()    
            results, dict_res, shortest = mobile_robot.calculate_trajectory(points_full, starting_configuration, prediction_horizon, vertex_angle_deg=40, magnitude_step=3, angle_step=3, loop=False)
            end_time = time.time()-start_time
            print(f"Time taken to compute sampling: {end_time}")
            print()

            for j in init_configs:
                ## CPC optimization'
                if args.optimization == "cpc":
                    cpc_plan = CPCPlanner(mobile_robot)
                    if j=='inverse_kinematic':
                        start_time = time.time()
                        X0_array, tf = initial_guess_simple(mobile_robot, points_full, starting_configuration, prediction_horizon, N)
                        initial_guess_time = time.time()-start_time
                        print(f"Time taken to compute initial guess inverse kynematic for scenario {i}: {initial_guess_time}")

                    elif j=='pontryagin':
                        start_time = time.time()
                        X0_array, tf = initial_guess(mobile_robot, points_full, starting_configuration, prediction_horizon, N)
                        # print(X0_array[100:150])
                        initial_guess_time = time.time()-start_time
                        print(f"Time taken to compute initial guess Pontryagin for scenario {i}: {initial_guess_time}")


                    X0 = ca.vertcat(tf, *X0_array)
                    start_time = time.time()
                    X, total_elements = cpc_plan.optimize(points_full, prediction_horizon, X0, np.inf, motion_model, forward_kinematic, N=N, obstacles_avoidance=obstacles_avoidance_for_opti)
                    optimization_time = time.time()-start_time
                    print(f"Time taken by CPC scenario {i}: {optimization_time}")
                    print(f'End time scenario {i}: {X[0]}')
                    ts = X[0]
                    Ns = [N]
                    dts = [float(ts/N)]
                    print()

                        

                ## Sequential optimization
                elif args.optimization == 'sequential':
                    sequential_plan = SequentialPlanner(mobile_robot)

                    if j=='inverse_kinematic':
                        start_time = time.time()
                        X0_array, ts_guess, Ns = initial_guess_simple(mobile_robot, points_full, starting_configuration, prediction_horizon, N, is_sequential_guess=True)
                        initial_guess_time = time.time()-start_time
                        print(f"Time taken to compute initial guess inverse kynematic for scenario {i}: {initial_guess_time}")

                    elif j=='pontryagin':
                        start_time = time.time()
                        X0_array, ts_guess, Ns = initial_guess(mobile_robot, points_full, starting_configuration, prediction_horizon, N, is_sequential_guess=True)
                        initial_guess_time = time.time()-start_time
                        print(f"Time taken to compute initial guess Pontryagin for scenario {i}: {initial_guess_time}")

                    X0 = ca.vertcat(*ts_guess, *X0_array)
                    start_time = time.time()
                    X, total_elements, ts = sequential_plan.optimize_sequential(points_full, prediction_horizon, X0, motion_model, forward_kinematic, Ns, obstacles_avoidance=obstacles_avoidance_for_opti)
                    optimization_time = time.time()-start_time
                    print(f"Time taken from Sequential scenario {i}: {optimization_time}")
                    print(f'End time scenario {i}: {X[0]}')
                    print()
                    X0 = ca.vertcat(sum(ts_guess), *X0_array)
                    dts = [float(ts[k]/Ns[k]) for k in range(len(Ns))]

                # data collection
                experiment_data['Scenario '+str(i)][j]["initial_guess_ts"].append(initial_guess_time)
                experiment_data['Scenario '+str(i)][j]["optimization_ts"].append(optimization_time)
                experiment_data['Scenario '+str(i)][j]["dt_std"].append(np.std(dts))
                experiment_data['Scenario '+str(i)][j]["total_elements"].append(total_elements)
                experiment_data['Scenario '+str(i)][j]["final_ts"].append(ts)
                experiment_data['Scenario '+str(i)][j]["final_Ns"].append(Ns)
                experiment_data['Scenario '+str(i)][j]["final_solutions"].append(X)
                experiment_data['Scenario '+str(i)][j]["initial_guesses"].append(X0)
        # Plot results per N
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
                                            experiment_data['Scenario '+str(i)][j]["final_ts"], experiment_data['Scenario '+str(i)][j]["final_Ns"], q_size=qs_num, v_bounds=v_bounds, a_bounds=a_bounds, labels=plotting_labels, show=False)
            tfs = [float(X[0]) for X in experiment_data['Scenario '+str(i)][j]["final_solutions"]]
            string = "Final times ->"
            for k in range(len(tfs)):
                string += f"{list_of_Ns[k]}: {tfs[k]:.3f}  "
            fig.suptitle(string)
            pdf.savefig(fig)
            plt.close(fig)
            # plot obstacle avoidance
            fig, obstacles_violated = plot_obstacle_avoidance(experiment_data['Scenario '+str(i)][j]["final_solutions"], obstacles_avoidance, experiment_data['Scenario '+str(i)][j]["total_elements"], show=False, labels=plotting_labels, q_size=qs_num)
            pdf.savefig(fig)
            plt.close(fig)
            # plot initialization times as N increases
            fig = simple_plot(list_of_Ns, [[experiment_data['Scenario '+str(i)][j]["initial_guess_ts"]]], y_label="Time", x_label="N", titles=[f'Initialization time for Scenario {i}'], show=False)
            pdf.savefig(fig)
            plt.close(fig)
            # plot optimization times as N increases
            fig = simple_plot(list_of_Ns, [[experiment_data['Scenario '+str(i)][j]["optimization_ts"]]], y_label="Time", x_label="N", titles=[f'Optimization time for Scenario {i}'], show=False)
            pdf.savefig(fig)
            plt.close(fig)
            # plot final times as N increases
            fig = simple_plot(list_of_Ns, [[[float(X[0]) for X in experiment_data['Scenario '+str(i)][j]["final_solutions"]]]], y_label="Time", x_label="N", show=False, titles=[f"Final time for Scenario {i}"])
            pdf.savefig(fig)
            plt.close(fig)

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
        for i in range(int(args.scenarios)):
            init_times.append(experiment_data['Scenario '+str(i)][init_method]["initial_guess_ts"])
            opti_times.append(experiment_data['Scenario '+str(i)][init_method]["optimization_ts"])
            labels.append(f"Scenario {i}")
            final_times.append([float(X[0]) for X in experiment_data['Scenario '+str(i)][init_method]["final_solutions"]])
        init_data.append(init_times)
        opti_data.append(opti_times)
        final_data.append(final_times)
    fig = simple_plot(list_of_Ns, init_data, x_label="N", y_label="Time", show=False, labels=labels, titles=[f"Initial guess time Pontryagin {args.optimization}", f"Initial guess time Inverse kinematic {args.optimization}"])
    # set title of the plot
    fig.suptitle("Overall graphs")
    pdf.savefig(fig)
    plt.close(fig)
    fig = simple_plot(list_of_Ns, opti_data, x_label="N", y_label="Time", show=False, labels=labels, titles=[f"Optimization time Pontryagin {args.optimization}", f"Optimization time Inverse kinematic {args.optimization}"])
    pdf.savefig(fig)
    plt.close(fig)
    # plot all final times
    fig = simple_plot(list_of_Ns, final_data, x_label="N", y_label="Time", show=False, labels=labels, titles=[f"Final time Pontryagin {args.optimization}", f"Final time Inverse kinematic {args.optimization}"])
    pdf.savefig(fig)
    plt.close(fig)
    # whisker plots of dts
    data = [] # will contain 2 lists, one for pontryagin and one for inverse kinematic
    for init_method in init_configs:
        dt_std = [] # is going to be a 4x7 array, but we want 7x4
        for i in range(int(args.scenarios)):
            dt_std.append(experiment_data['Scenario '+str(i)][init_method]["dt_std"]) # will contain a list of stds (as many as the number of Ns) per scenario
        dt_std = np.array(dt_std).T.tolist()
        data.append(dt_std)
    fig = plot_whisker_plots(data, plotting_labels, show=False, titles=["Standard dev of dts with Pontryagin", "Standard dev of dts with Inverse kinematic"], ylabel="Standard deviation of dts")
    pdf.savefig(fig)
    plt.close(fig)


    # close pdf
    pdf.close()






            


    
if __name__ == "__main__":
    main()