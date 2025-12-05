import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from mm_utils import parsing
from mm_utils.plotting import construct_logger

STYLE_PATH = parsing.parse_ros_path(
    {"package": "mm_utils", "path": "scripts/plot_style.mplstyle"}
)
plt.style.use(STYLE_PATH)
plt.rcParams["pdf.fonttype"] = 42


EE_POS_Target0 = np.array([1.5, -2.0])
EE_POS_Target1 = np.array([3.5, -1.0])  # home position + 4m in x direciton
Rc = 0.5 + 0.25  # robot radius + padding
YELLOW = "#FFAE42"
SCARLET = "#cc1d00"


def closet_approaching_pose(curr_base_pos, curr_ee_pos, next_ee_pos, rc):
    # line segment a that connecting curr_base_pos with next_ee_pos
    a = next_ee_pos - curr_base_pos
    a_hat = a / np.linalg.norm(a)

    n_hat = np.cross([0, 0, 1], np.hstack((a_hat, 0)))

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
    cm_index = [0.8, 0.6, 0.4, 0.2]
    line_width = [1.5, 2.0, 1.0, 1.0, 1.0]
    colors = sns.color_palette("ch:s=.35,rot=-.35", n_colors=3)
    cm = LinearSegmentedColormap.from_list("my_list", colors, N=50)

    # plot base path
    for id, p in enumerate(plotters):
        if p.data["name"][:8] == "Baseline":
            sns.color_palette("Paired")
            color = SCARLET
        else:
            color = cm(cm_index[id])

        axes = p.plot_base_path(
            axes, id, worldframe=False, linewidth=line_width[id], color=color
        )

    # plot base ref path
    plotters[-1].plot_base_ref_path(
        axes, num_plotters, legend="", worldframe=False, color=cm(cm_index[1])
    )

    r_ew_w_ds = plotters[-1].data.get("r_ew_b_ds", [])
    r_bw_w_ds = plotters[-1].data.get("r_bw_b_ds", [])
    base_ref_path_text_pos = (r_bw_w_ds[0] + r_bw_w_ds[-1]) / 2
    base_ref_path_text_pos += np.array([0, 0.02])
    axes.text(
        base_ref_path_text_pos[0],
        base_ref_path_text_pos[1],
        "Base Reference Path",
        horizontalalignment="center",
    )
    # plot EE targets
    axes.scatter(
        EE_POS_Target0[0], EE_POS_Target0[1], edgecolor=YELLOW, facecolor=YELLOW
    )
    axes.scatter(EE_POS_Target1[0], EE_POS_Target1[1], color=YELLOW)
    axes.text(
        EE_POS_Target0[0] + 0.35,
        EE_POS_Target0[1],
        "EE \nWaypoint \n#1",
        horizontalalignment="center",
    )
    axes.text(
        EE_POS_Target1[0] - 0.25,
        EE_POS_Target1[1] + 0.1,
        "EE Waypoint \n#2",
        horizontalalignment="center",
    )

    #
    circle1 = plt.Circle(
        r_ew_w_ds[0][:2], Rc, facecolor=[0, 0, 0, 0.015], edgecolor=cm(cm_index[0])
    )
    axes.add_patch(circle1)

    circle_text_phi = 0.05 * np.pi
    circle_text_pos = (
        EE_POS_Target0
        + Rc * np.array([np.cos(circle_text_phi), np.sin(circle_text_phi)]) * 1.8
    )
    axes.text(
        circle_text_pos[0],
        circle_text_pos[1],
        "Closest\nApproaching Circle",
        horizontalalignment="center",
    )

    approach_pose = closet_approaching_pose(
        np.zeros(2), EE_POS_Target0[:2], EE_POS_Target1[:2], Rc
    )

    arrow_length = 0.2
    axes.arrow(
        approach_pose[0],
        approach_pose[1],
        arrow_length * np.cos(approach_pose[2]),
        arrow_length * np.sin(approach_pose[2]),
        width=0.01,
        facecolor=SCARLET,
        edgecolor=SCARLET,
    )
    closet_approaching_pose_text_pos = (
        approach_pose[:2]
        + np.array([np.cos(approach_pose[2]), np.sin(approach_pose[2])])
        * arrow_length
        * 1.75
    )
    axes.text(
        closet_approaching_pose_text_pos[0],
        closet_approaching_pose_text_pos[1],
        "Intermediate\nBase Waypoint",
    )
    axes.set_ylim([-2.1, 0.25])

    axes.set_aspect("equal")
    plt.legend(loc="lower left")
    plt.grid("on")

    plt.subplots_adjust(
        top=0.99, bottom=0.166, left=0.083, right=0.974, hspace=0.15, wspace=0.16
    )

    f = plt.gcf()
    f.savefig(folder_path + "/path_comparison.pdf", pad_inches=0)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", required=True, help="Path to data folder.")
    parser.add_argument("--compare", action="store_true", help="plot comparisons")
    args = parser.parse_args()

    path_comparison(args.folder)
