import argparse
import os

import pandas
import numpy as np

from mmseq_utils.logging import DataLogger, DataPlotter, multipage, construct_logger
from mmseq_utils import math
import matplotlib.pyplot as plt

EE_POS_Target0 = np.array([1.5, -2.0])
EE_POS_Target1 = np.array([4., 0.])     # home position + 4m in x direciton
Rc = 0.5 + 0.25         # robot radius + padding

def closet_approaching_pose(curr_base_pos, curr_ee_pos, next_ee_pos, rc):
    # line segment a that connecting curr_base_pos with next_ee_pos
    a = (next_ee_pos - curr_base_pos)
    a_hat = a / np.linalg.norm(a)

    n_hat = np.cross([0,0,1], np.hstack((a_hat, 0)))

    approach_point = curr_ee_pos + n_hat[:2] * rc
    approch_heading = np.arctan2(a[1], a[0])

    return np.hstack((approach_point, approch_heading))

def path_comparison(folder_path):
    plotters = []
    for filename in sorted(os.listdir(folder_path)):
        d = os.path.join(folder_path, filename)
        if os.path.isdir(d):
            plotter = construct_logger(d)
            plotters.append(plotter)

    num_plotters = len(plotters)
    axes = None
    for id, p in enumerate(plotters):
        axes = p.plot_base_path(axes, id)

    plotters[0].plot_base_ref_path(axes, num_plotters, legend="")
    plotters[0].plot_ee_waypoints(axes, num_plotters, legend="")

    approach_pose = closet_approaching_pose(np.zeros(2), EE_POS_Target0, EE_POS_Target1, Rc)
    print(approach_pose)

    arrow_length = 0.2
    axes.arrow(approach_pose[0], approach_pose[1],
               arrow_length*np.cos(approach_pose[2]), arrow_length*np.sin(approach_pose[2]), width=0.01, facecolor='r', edgecolor='r')

    circle1 = plt.Circle(EE_POS_Target0, Rc, facecolor=[1,0,0,0.25], edgecolor='r')
    axes.add_patch(circle1)
    axes.set_aspect("equal")
    plt.legend()



    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", required=True, help="Path to data folder.")
    parser.add_argument("--compare", action="store_true",
                        help="plot comparisons")
    args = parser.parse_args()

    path_comparison(args.folder)