"""Plot MPC logging data from PyBullet simulations."""

import argparse

import matplotlib.pyplot as plt

from mm_utils.plotting import construct_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PyBullet simulation data")
    parser.add_argument("--folder", required=True, help="Path to data folder")
    parser.add_argument("--mpc", action="store_true", help="Plot MPC info")
    parser.add_argument("--tracking", action="store_true", help="Plot tracking data")
    parser.add_argument("--robot", action="store_true", help="Plot robot data")
    parser.add_argument("--all", action="store_true", help="Plot all data")

    args = parser.parse_args()
    plotter = construct_logger(args.folder)

    if args.all:
        plotter.plot_all()
    else:
        if args.mpc:
            plotter.plot_mpc()
        if args.tracking:
            plotter.plot_tracking()
        if args.robot:
            plotter.plot_robot()

    plt.show()
