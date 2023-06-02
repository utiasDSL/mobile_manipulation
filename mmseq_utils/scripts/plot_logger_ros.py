#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
from mmseq_utils.logging import DataLogger, DataPlotter
import matplotlib.pyplot as plt

def plot_tracking(plotters):
    # ee_axes = plotters["control"].plot_ee_position()
    # base_axes = plotters["control"].plot_base_position()
    # plotters["sim"].plot_ee_position(ee_axes)
    # plotters["sim"].plot_base_position(base_axes)
    plotters.plot_ee_position()
    plotters.plot_base_position()
    plotters.plot_tracking_err()

def construct_logger(path_to_folder):
    return DataPlotter.from_ROS_results(path_to_folder)
    # data = {}
    # for filename in os.listdir(path_to_folder):
    #     d = os.path.join(path_to_folder, filename)
    #     key = filename.split("_")[0]
    #     if os.path.isdir(d):
    #         path_to_npz = os.path.join(d, "data.npz")
    #         data[key] = DataPlotter.from_npz(path_to_npz)
    #
    # return data

def plot_comparisons(args):
    plotters = []
    for filename in os.listdir(args.folder):
        d = os.path.join(args.folder, filename)
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

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", required=True, help="Path to data folder.")
    parser.add_argument("--htmpc", action="store_true", help="plot htmpc info")
    parser.add_argument("--tracking", action="store_true",
                        help="plot tracking data")
    parser.add_argument("--robot", action="store_true",
                        help="plot tracking data")

    parser.add_argument("--compare", action="store_true",
                        help="plot comparisons")
    args = parser.parse_args()

    if args.compare:
        plot_comparisons(args)
    else:
        data_plotter = construct_logger(args.folder)
        if args.htmpc:
            data_plotter.plot_mpc()
        if args.tracking:
            plot_tracking(data_plotter)
            data_plotter.plot_task_violation()
        if args.robot:
            data_plotter.plot_robot()

        plt.show()