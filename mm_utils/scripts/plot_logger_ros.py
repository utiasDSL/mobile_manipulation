"""Plot MPC logging data from ROS experiments."""

import argparse
import os

import matplotlib.pyplot as plt
from pandas import DataFrame

from mm_utils.plotting import DataPlotter, construct_logger, multipage

STATS = [
    ("err_ee", "integral"),
    ("err_ee", "rms"),
    ("err_base_normalized", "integral"),
    ("err_base_normalized", "rms"),
    ("cmd_accs_saturation", "mean"),
    ("run_time", "mean"),
    ("constraints_violation", "mean"),
]


def statistics(plotters):
    """Compute statistics from a list of plotters."""
    stats_dict = {}
    for p in plotters:
        stats = p.summary(STATS)
        stats_dict[p.data["name"]] = stats

    df = DataFrame.from_dict(
        stats_dict, orient="index", columns=[p[0] + "_" + p[1] for p in STATS]
    )
    return df


def plot_comparisons(folder_path, show=True):
    """Plot comparison of multiple experiment results."""
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

    plt.close("all")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MPC logging data")
    parser.add_argument("-f", "--folder", required=True, help="Path to data folder")
    parser.add_argument("--control", action="store_true", help="Plot MPC control info")
    parser.add_argument("--show", action="store_true", help="Show plots")
    parser.add_argument("--tracking", action="store_true", help="Plot tracking data")
    parser.add_argument("--robot", action="store_true", help="Plot robot data")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check collision, constraint violation, tracking",
    )
    parser.add_argument("--sdf", action="store_true", help="Plot SDF map")
    parser.add_argument("--runmpc", action="store_true", help="Rerun MPC iterations")
    parser.add_argument("--timestep", type=float, help="Control timestep to inspect")
    parser.add_argument("--sdfdebug", action="store_true", help="Debug SDF")
    parser.add_argument("--mpcdebug", action="store_true", help="Debug MPC constraints")
    parser.add_argument("--compare", action="store_true", help="Plot comparisons")
    parser.add_argument(
        "--savefigs", action="store_true", help="Export figures to file"
    )
    parser.add_argument(
        "--printparam", action="store_true", help="Print MPC parameters"
    )
    parser.add_argument("--paramname", type=str, help="Parameter name to print")
    parser.add_argument(
        "--plottimeoptimal", action="store_true", help="Plot time optimal plan tracking"
    )
    parser.add_argument("--exp", action="store_true", help="Plot experiment results")

    args = parser.parse_args()

    if args.compare:
        plot_comparisons(args.folder, args.show)
    else:
        data_plotter = construct_logger(args.folder, data_plotter_class=DataPlotter)
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
