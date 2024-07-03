#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
from mmseq_utils.logging import DataLogger
from mmseq_utils.plotting import DataPlotter
import matplotlib.pyplot as plt

def plot_tracking(plotters):
    ee_axes = plotters["control"].plot_ee_position()
    base_axes = plotters["control"].plot_base_position()
    plotters["sim"].plot_ee_position(ee_axes)
    plotters["sim"].plot_base_position(base_axes)

def construct_logger(path_to_folder):
    plotter = DataPlotter.from_PYSIM_results(path_to_folder)

    return plotter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Path to data folder.")
    parser.add_argument("--htmpc", action="store_true", help="plot htmpc info")
    parser.add_argument("--tracking", action="store_true",
                        help="plot tracking data")
    parser.add_argument("--robot", action="store_true",
                        help="plot tracking data")

    parser.add_argument("--all", action="store_true",
                        help="plot all")

    args = parser.parse_args()
    plotter = construct_logger(args.folder)
    if args.all:
        plotter.plot_all()
    else:
        if args.htmpc:
            plotter.plot_mpc()
        if args.tracking:
            plotter.plot_tracking()
        if args.robot:
            plotter.plot_robot()

    plt.show()