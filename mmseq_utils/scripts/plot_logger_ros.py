#!/home/tracy/Projects/mm_catkin_ws/src/venv/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os

import pandas

from mmseq_utils.logging import DataLogger
from mmseq_utils.plotting import DataPlotter, multipage, construct_logger, HTMPCDataPlotter
import matplotlib.pyplot as plt
from pandas import DataFrame, concat

STATS = [("err_ee", "integral"), ("err_ee", "rms"), ("err_base_normalized", "integral"),
         ("err_base_normalized", "rms"),
         ("cmd_accs_saturation", "mean"), ("run_time", "mean"), ("constraints_violation", "mean")]

def statistics(plotters):

    stats_dict = {}
    for id, p in enumerate(plotters):
        stats = p.summary(STATS)
        stats_dict[p.data["name"]] = stats

    df = DataFrame.from_dict(stats_dict, orient='index', columns=[p[0] +"_"+ p[1]for p in STATS])

    return df

def plot_comparisons(folder_path, show=True):
    plotters = []
    for filename in os.listdir(folder_path):
        d = os.path.join(folder_path, filename)
        if os.path.isdir(d):
            plotter = construct_logger(d)
            plotters.append(plotter)

    # plot tracking error
    axes = None
    for id, p in enumerate(plotters):
        axes = p.plot_tracking_err(axes, id)

    # plot commanded velocity and acceleration
    axes = None
    for id, p in enumerate(plotters):
        axes = p.plot_cmds(axes, id)

    # plot Task performance
    axes = None
    for id, p in enumerate(plotters):
        axes = p.plot_task_performance(axes, id)

    multipage(os.path.join(folder_path, "plots.pdf"), dpi=163)
    df = statistics(plotters)
    df.to_csv(os.path.join(folder_path, "stats.csv"))
    if show:
        plt.show()

    plt.close('all')
    return df

def plot_comparisons_idkc(folder_path, show=True):
    plotters = []
    for filename in os.listdir(folder_path):
        if filename.split("_")[0] != "HTIDKC":
            continue
        else:
            print(filename)
            d = os.path.join(folder_path, filename)
            if os.path.isdir(d):
                plotter = construct_logger(d)
                plotters.append(plotter)

    # plot tracking error
    axes = None
    for id, p in enumerate(plotters):
        axes = p.plot_task_violation(axes, id)

    if show:
        plt.show()

def benchmark(folder_path):
    """

    :param folder_path: path to benchmark data folder
    :return:
    """
    # Generate statistics in each test result folder
    # NOTE: Decided not to generate plots here. It takes too much memory space. Generate plots using bash scripts instead.
    if "stats_all.csv" in os.listdir(folder_path):
        df = pandas.read_csv(os.path.join(folder_path, "stats_all.csv"), index_col=[0,1])
        print(df)
    else:
        dfs = []
        keys = []
        for folder_name in os.listdir(folder_path):
            test_folder_path = os.path.join(folder_path, folder_name)
            if os.path.isdir(test_folder_path):
                plotters = []
                for filename in os.listdir(test_folder_path):
                    d = os.path.join(test_folder_path, filename)
                    if os.path.isdir(d):
                        plotter = construct_logger(d)
                        plotters.append(plotter)

                keys.append(folder_name)
                dfs.append(statistics(plotters))

        # concatenate data frames
        df = concat(dfs, keys=keys)
        df.to_csv(os.path.join(folder_path, "stats_all.csv"))

    df_mpc = df.xs("HTMPC", level=1)
    df_idkc = df.xs("HTIDKC", level=1)
    df_summary = concat([df_mpc.mean(), df_idkc.mean()], keys=["HTMPC", "HTIDKC"], axis=1)
    df_summary.to_csv(os.path.join(folder_path, "stats_summary.csv"))

    diff_ee = (df_mpc["err_ee_integral"] - df_idkc["err_ee_integral"]) / df_idkc["err_ee_integral"] * 100
    diff_base = (df_mpc["err_base_normalized_integral"] - df_idkc["err_base_normalized_integral"]) / df_idkc["err_base_normalized_integral"] * 100
    dJ = [diff_ee.to_numpy() , diff_base.to_numpy()]

    f, axes = plt.subplots(1,1)
    plt.scatter(dJ[0], dJ[1])
    axes.set_xlabel("ΔJ_1 (%)")
    axes.set_ylabel("ΔJ_2 (%)")
    axes.set_title("HTMPC vs HTIDKC on Lexicographic Optimality")

    for (key, val) in STATS:
        name = key + "_" + val
        f = plt.figure()
        plt.hist(df_idkc[name], bins=100, label="HTIDKC", alpha=0.5)
        plt.hist(df_mpc[name], bins=100, label="HTMPC", alpha=0.5)
        plt.legend()
        plt.title(name)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", required=True, help="Path to data folder.")
    parser.add_argument("--control", action="store_true", help="plot htmpc info")
    parser.add_argument("--show", action="store_true", help="plot htmpc info")
    parser.add_argument("--tracking", action="store_true",
                        help="plot tracking data")
    parser.add_argument("--robot", action="store_true",
                        help="plot tracking data")
    parser.add_argument("--check", action="store_true",
                        help="check collision, constraint violation, tracking")
    parser.add_argument("--sdf", action="store_true",
                        help="plot sdf")
    parser.add_argument("--runmpc", action="store_true", help="rerun mpc iters")
    parser.add_argument("--timestep", type=float,help="control timestep to inspect")
    parser.add_argument("--sdfdebug", action="store_true",
                        help="debug")
    parser.add_argument("--mpcdebug", action="store_true",
                        help="debug")
    parser.add_argument("--compare", action="store_true",
                        help="plot comparisons")
    parser.add_argument("--htmpc", action="store_true", default=False,
                        help="use htmpc plotter")
    parser.add_argument("--compareidkc", action="store_true",
                        help="plot comparisons")
    parser.add_argument("--benchmark", action="store_true", help="gather benchmark results")
    parser.add_argument("--savefigs", action="store_true", help="export figures to file")
    parser.add_argument("--printparam", action="store_true", help="export figures to file")
    parser.add_argument("--paramname", type=str,help="control timestep to inspect")
    parser.add_argument("--plottimeoptimal", action="store_true", help="export figures to file")
    parser.add_argument("--exp", action="store_true", help="export figures to file")

    args = parser.parse_args()

    if args.compare:
        plot_comparisons(args.folder, args.show)
    if args.compareidkc:
        plot_comparisons_idkc(args.folder, args.show)
    elif args.benchmark:
        benchmark(args.folder)
    else:
        if args.htmpc:
            plotter_class = HTMPCDataPlotter
        else:
            plotter_class = DataPlotter
        data_plotter = construct_logger(args.folder, is_htmpc=args.htmpc, data_plotter_class=plotter_class)
        if args.control:
            data_plotter.plot_mpc()
        if args.sdf:
            data_plotter.plot_sdf_map(args.timestep)
        if args.tracking:
            data_plotter.plot_tracking()
        if args.robot:
            data_plotter.plot_robot()
        if args.check:
            data_plotter.plot_quick_check()
        if args.sdfdebug:
            data_plotter.sdf_failure_debug(args.timestep)
        if args.mpcdebug:
            data_plotter.mpc_constraint_debug(args.timestep)
        if args.runmpc:
            data_plotter.run_mpc_iter(args.timestep)
        if args.printparam:
            data_plotter.print_mpc_param(args.paramname)
        if args.plottimeoptimal:
            data_plotter.plot_time_optimal_plan_tracking_results()
        if args.exp:
            data_plotter.plot_exp(args.timestep)
        if args.savefigs:
            data_plotter.save_figs()
        plt.show()



