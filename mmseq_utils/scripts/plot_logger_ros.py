#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
from mmseq_utils.logging import DataLogger, DataPlotter
import matplotlib.pyplot as plt

def plot_tracking(plotters):
    ee_axes = plotters["control"].plot_ee_position()
    base_axes = plotters["control"].plot_base_position()
    plotters["sim"].plot_ee_position(ee_axes)
    plotters["sim"].plot_base_position(base_axes)

def plot_cmd_vel(plotters):
    axes = plotters["control"].plot_cmds(legend="controller_")
    plotters["sim"].plot_cmds(axes=axes, legend="sim_")
    plotters["control"].plot_du()

def construct_logger(path_to_folder):
    data = {}
    for filename in os.listdir(path_to_folder):
        d = os.path.join(path_to_folder, filename)
        key = filename.split("_")[0]
        if os.path.isdir(d):
            path_to_npz = os.path.join(d, "data.npz")
            data[key] = DataPlotter.from_npz(path_to_npz)

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Path to data folder.")
    parser.add_argument("--htmpc", action="store_true", help="plot htmpc info")
    parser.add_argument("--tracking", action="store_true",
                        help="plot tracking data")
    parser.add_argument("--robot", action="store_true",
                        help="plot tracking data")

    args = parser.parse_args()
    data_plotter_dict = construct_logger(args.folder)
    if args.htmpc:
        data_plotter_dict["control"].plot_mpc()
    if args.tracking:
        plot_tracking(data_plotter_dict)
    if args.robot:
        data_plotter_dict["sim"].plot_robot()
        plot_cmd_vel(data_plotter_dict)

    plt.show()