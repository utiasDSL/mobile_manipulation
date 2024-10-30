import argparse
import os
from typing import Optional, List, Dict, Tuple, Union

import pandas
import numpy as np

from mmseq_utils.plotting import construct_logger, DataPlotter, HTMPCDataPlotter
from mmseq_utils import parsing
from mmseq_utils.matplotlib_helper import plot_square_box, plot_circle, plot_cross
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
STYLE_PATH = parsing.parse_ros_path({"package": "mmseq_utils", "path":"scripts/plot_style.mplstyle"})
plt.style.use(STYLE_PATH)
plt.rcParams['pdf.fonttype'] = 42


BOX_CENTER = [4.8, -0.3, 0.3]
BASE_RADIUS = 0.55
YELLOW="#FFAE42"
SCARLET="#cc1d00"


class SemiStaticDataPlotter(DataPlotter):
    def __init__(self, data, config=None, process=True):
        super().__init__(data, config, process)
        if process:
            self.trim_data()
            self.get_key_time_index()

    def trim_data(self):
        x_cutoff = 5.5
        rb_x = self.data["raw"]['xs']['value'][:, 0]

        stop_index_raw = np.where(rb_x > x_cutoff)[0]
        if len(stop_index_raw) > 0:
            stop_index_raw = np.where(rb_x > x_cutoff)[0][0]
            stop_time = self.data["raw"]['xs']['ts'][stop_index_raw]-1
            stop_index = np.where(self.data['ts']>stop_time)[0][-1]
        else:
            stop_index = np.where(self.data["signed_distance_sdf"] < 0)[0]
            if len(stop_index) > 0:
                stop_index = stop_index[0] + 1
                stop_time = self.data['ts'][stop_index]
                stop_index_raw = np.where(self.data["raw"]['xs']["ts"] < stop_time)[0][-1]

        for key, value in self.data["raw"].items():
            value["ts"] = value['ts'][:stop_index_raw]
            value["value"] = value["value"][:stop_index_raw]
        
        for key, value in self.data.items():
            if key =="raw" or not isinstance(value, np.ndarray) or value.ndim == 0:
                continue
            else:
                self.data[key] = self.data[key][:stop_index]
    
    def get_key_time_index(self):
        
        print(self.data["signed_distance_sdf"][:10])
        self.data["t_idx_map_available"] = np.where(np.abs(self.data["signed_distance_sdf"] - self.data["signed_distance_sdf"][0]) > 0.1)[0][0]
        self.data["t_idx_first_control_available"] = self.data["t_idx_map_available"] + 1


    def plot_base_path(self, axes=None, index=0, legend=None, worldframe=True, linewidth=1, color=None, linestyle="-"):
        r_b_dict = self.data["raw"].get("r_bw_ws", None)

        if r_b_dict is None:
            return
        else:
            r_b = r_b_dict["value"]

        if axes is None:
            axes = []
            f, axes = plt.subplots(1, 1, sharex=True)

        if legend is None:
            legend = self.data["name"]

        # if legend == "baseline h = 0.8m":
        #     r_ew_ws += [0, 1]
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        # colors = sns.color_palette("ch:s=.35,rot=-.35", n_colors=3)
        cm = LinearSegmentedColormap.from_list(
            "my_list", colors, N=50)
        if color is None:
            color = cm(index)
        if len(r_b) > 0:
            axes.plot(r_b[:, 0], r_b[:, 1], label=legend, color=color, linewidth=linewidth, linestyle=linestyle)


        axes.grid()
        axes.legend()
        axes.set_xlabel("x (m)")
        axes.set_ylabel("y (m)")
        axes.set_title("Base Path Tracking")

        return axes

    def plot_robot_boundary(self, axes):
        r_b = self.data.get("r_bw_ws", None)

        if r_b is None:
            return
        
        collision_time_index = np.where(self.data["signed_distance"]<0)[0]
        if len(collision_time_index) > 0:
            plot_circle(r_b[collision_time_index[0], :2], BASE_RADIUS, axes)
            plot_cross(r_b[collision_time_index[0], :2],linewidth=2, length=0.05, ax=axes)
    
    def plot_key_base_frame(self, axes, color):
        r_b = self.data.get("r_bw_ws", None)

        r_b_map_available = r_b[self.data["t_idx_map_available"]][:2]
        r_b_first_control_available = r_b[self.data["t_idx_first_control_available"]][:2]

        axes.scatter(r_b_map_available[0], r_b_map_available[1] , color='r')
        axes.scatter(r_b_first_control_available[0], r_b_first_control_available[1], color='g')

        # x_bar = self.data["mpc_x_bars"][self.data["t_idx_map_available"]]
        # axes.scatter(x_bar[:, 0], x_bar[:, 1], s=2, color=color, facecolor=None, linewidth=2)

    def plot_sdf_collision(self, axes=None, index=0, color=None, linewidth=1, block=True, legend=None):
        nq = int(self.data["nq"])
        ts_downsampled = self.data["ts"]
        ts = self.data["raw"]["xs"]["ts"]
        qs = self.data["raw"]["xs"]["value"][:, :nq]
        names = ["self", "static_obstacles"]
        params = {"self": [], "static_obstacles":[]}

        if self.config["controller"]["sdf_collision_avoidance_enabled"]:
            param_names =  ["_".join(["mpc","sdf", "param", str(i)])+"s" for i in range(self.model_interface.sdf_map.dim+1)]
            sdf_params = [self.data[name] for name in param_names]
            sdf_params_resampled = []

            for param in sdf_params:
                interpolator = interp1d(ts_downsampled, param, kind='nearest', fill_value="extrapolate", axis=0)
                sdf_params_resampled.append(interpolator(ts))
            params["sdf"] = sdf_params_resampled
            names += ["sdf"]

        sds = self.model_interface.evaluteSignedDistance(names, qs, params)
        sds = sds["sdf"]

        if axes is None:
            f, axes = plt.subplots(1, 1, sharex=True)

        if legend is None:
            legend = self.data["name"]

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        # colors = sns.color_palette("ch:s=.35,rot=-.35", n_colors=3)
        cm = LinearSegmentedColormap.from_list(
            "my_list", colors, N=50)

        
        if color is None:
            color = cm(index)

        axes.plot(ts, sds, label=legend, color=color, linestyle="-",linewidth=linewidth)
        axes.set_ylabel("Sd(q) (m)")
        axes.set_xlabel("t (s)")

        axes.legend()
        axes.grid('on')

        return axes
    
class SemiStaticHTDataPlotter(HTMPCDataPlotter):

    def __init__(self, folder_path, data_plotter_class=SemiStaticDataPlotter):
        super().__init__(folder_path, data_plotter_class)
        for plotter in self.plotters:
            plotter.trim_data()
    
    def plot_robot_boundary(self, axes):
        self.plotters[-1].plot_robot_boundary(axes)

    def plot_key_base_frame(self, axes, color):
        self.plotters[-1].plot_key_base_frame(axes, color)

    def plot_sdf_collision(self, axes=None, index=0, color=None, linewidth=1, block=True, legend=None):
        axes = self.plotters[-1].plot_sdf_collision(axes, index, color, linewidth, block, legend)
        return axes

    def plot_base_path(self, axes=None, index=0, legend=None, worldframe=True, linewidth=1, color=None,linestyle='-'):
        axes = self.plotters[-1].plot_base_path(axes, index, legend, worldframe, linewidth, color, linestyle)
        return axes

class SemiStaticBenchmarkPlotter():
    plotters: List[Union[DataPlotter, SemiStaticDataPlotter, SemiStaticHTDataPlotter]]

    def __init__(self, args) -> None:
        folder_path = args.folder
        self.plotters = []
        for filename in sorted(os.listdir(folder_path)):
            d = os.path.join(folder_path, filename)
            if os.path.isdir(d):
                is_htmpc = "HTMPC" in filename
                if is_htmpc:
                    data_plotter_class =SemiStaticHTDataPlotter
                    ht_data_plotter_class=SemiStaticDataPlotter
                else:
                    data_plotter_class=SemiStaticDataPlotter
                    ht_data_plotter_class = SemiStaticDataPlotter # this doesn't matter

                plotter = construct_logger(d, is_htmpc=is_htmpc, data_plotter_class=data_plotter_class, 
                                        ht_data_plotter_class=ht_data_plotter_class)
                # plotter = construct_logger(d, is_htmpc=True)

                self.plotters.append(plotter)

        self.num_plotters = len(self.plotters)
        self.folder_path = folder_path
        try:
            self.perception_distance = float(folder_path.split('/')[-1][:3])
        except ValueError:
            self.perception_distance = 2.4

        self.t_idx_perceptive_raw = np.where(self.plotters[0].data["raw"]["xs"]["value"][:, 0] < BOX_CENTER[0] - self.perception_distance - 0.3)[0][-1]

    def plot_base_path(self):

        f, axes = plt.subplots(1, 1, sharex=True, figsize=(3.5, 1.3))
        cm_index = [0.8, 0.6, 0.4, 0.2]
        line_width = [2.0, 2.0, 2.0, 2.0, 2.0]
        colors = sns.color_palette("ch:s=.35,rot=-.35", n_colors=3)
        cm = LinearSegmentedColormap.from_list(
            "my_list", colors, N=50)
        
        # plot perceptive region
        x_perceptive = BOX_CENTER[0] - 0.3 - self.perception_distance
        axes.axvspan(x_perceptive, 7, color="green", alpha=0.1)  # alpha for transparency
        axes.axvline(x_perceptive, color='darkgreen', linestyle='--')  # Optional visual for x-value

        arrow_length = 1.8
        y = -0.5
        axes.annotate(
            '', 
            xy=(x_perceptive + arrow_length, y), 
            xytext=(x_perceptive+arrow_length*0.05, y),
            arrowprops=dict(arrowstyle='->', linewidth=2)
        )

        # Plot left-pointing arrow
        axes.annotate(
            '', 
            xy=(x_perceptive - arrow_length, y), 
            xytext=(x_perceptive-arrow_length*0.05, y),
            arrowprops=dict(arrowstyle='->', linewidth=2)
        )

        # Add text annotation at the center
        axes.text(
            x_perceptive+arrow_length/2, y, "Perceptive",
            ha='center', va='center', 
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
        )
        axes.text(
            x_perceptive-arrow_length/2, y, "Blinded",
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
        )

        # plot base path
        plot_square_box(0.6, BOX_CENTER[:2], axes)

        for id, p in enumerate(self.plotters):
            if p.data["name"][:8] == "Baseline":
                sns_color= sns.color_palette("Paired")
                color = SCARLET
            else:
                color = cm(cm_index[id])
            axes = p.plot_base_path(axes, id, worldframe=False, linewidth=line_width[id], color=color)
            p.plot_robot_boundary(axes)
            # p.plot_key_base_frame(axes, color=color)

        # plot base ref path
        self.plotters[-1].plot_base_ref_path(axes, self.num_plotters, legend="ref", worldframe=False, color=SCARLET)
        
        # plotters[0].plot_ee_waypoints(axes, num_plotters, legend="")

        # axes.set_aspect("equal")
        plt.legend(loc="upper left")
        plt.grid('on')

        axes.set_aspect('equal', 'box')
        axes.set_title('Base Path')
        axes.set_xlim([-0.1, 6.1])
        axes.annotate("Base collided \nwith the box.", xy=(4.5, -0.05), xycoords="data",
                            xytext=(5.5, 0.4),
                            horizontalalignment='center', arrowprops=dict(arrowstyle="->", lw=1.),
                            fontsize=6,
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))



        plt.subplots_adjust(top=0.92,
                            bottom=0.218,
                            left=0.152,
                            right=0.974,
                            hspace=0.2,
                            wspace=0.2)



        f = plt.gcf()
        f.savefig(self.folder_path + "/path_comparison.pdf" , pad_inches=0)

    def plot_base_path_htmpc_vs_stmpc(self):

        f, axes = plt.subplots(1, 1, sharex=True, figsize=(3.5, 1.3))
        cm_index = [0.8, 0.6, 0.4, 0.2]
        line_width = [2.0, 2.0, 2.0, 2.0, 2.0]
        colors = sns.color_palette("ch:s=.35,rot=-.35", n_colors=3)
        cm = LinearSegmentedColormap.from_list(
            "my_list", colors, N=50)
        
        # plot base path
        for id, p in enumerate(self.plotters):
            if p.data["name"][:8] == "Baseline":
                sns_color= sns.color_palette("Paired")
                color = SCARLET
            else:
                color = cm(cm_index[id%3])
            linestyle = "-." if "HTMPC" in p.data['name'] else '-'
            axes = p.plot_base_path(axes, id, worldframe=False, linewidth=line_width[id%3], color=color, linestyle=linestyle, legend='')
            p.plot_robot_boundary(axes)
            # p.plot_key_base_frame(axes, color=color)


        # plot base ref path
        self.plotters[-1].plot_base_ref_path(axes, self.num_plotters, legend="ref", worldframe=False, color=SCARLET)
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0],color=cm(0.8), lw=1, linestyle='-', label="STMPC-CBF"),
            Line2D([0], [0],color=cm(0.8), lw=1, linestyle='-.', label="HTMPC-CBF"),
        ]

        # Add the legend
        main_legend = plt.legend(handles=legend_elements, title="Controller Type",loc="upper left")
        plt.legend(handles=[Line2D([0], [0],color=SCARLET, lw=2, linestyle='--', label="Ref")],loc="lower right")
        axes.add_artist(main_legend)

        # Annotate Gamma
        arrow_end_x = 5.5
        arrow_start_x = 6.5
        text_x = arrow_start_x
        axes.annotate("", xy=(arrow_end_x, 1.15), xycoords="data",
                            xytext=(arrow_start_x, 1.15),
                            horizontalalignment='center', arrowprops=dict(arrowstyle="->", lw=1.),
                            fontsize=6,
                            bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round,pad=0.3'))
        axes.text(
            text_x, 1.15, "γ = 0.1",
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round,pad=0.3')
        )
        axes.annotate("", xy=(arrow_end_x, 0.8), xycoords="data",
                            xytext=(arrow_start_x, 0.8),
                            horizontalalignment='center', arrowprops=dict(arrowstyle="->", lw=1.),
                            fontsize=6,
                            bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round,pad=0.3'))
        axes.text(
            text_x, 0.8, "γ = 0.5",
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round,pad=0.3')
        )
        axes.annotate("", xy=(arrow_end_x, 0.64), xycoords="data",
                            xytext=(arrow_start_x, 0.45),
                            horizontalalignment='center', arrowprops=dict(arrowstyle="->", lw=1.),
                            fontsize=6,
                            bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round,pad=0.3'))
        axes.text(
            text_x, 0.45, "γ = 0.9",
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round,pad=0.3')
        )
        plt.grid('on')

        axes.set_aspect('equal', 'box')
        axes.set_title('Base Path')
        axes.set_xlim([1.5 , 7.5])

        plt.subplots_adjust(top=0.92,
                            bottom=0.218,
                            left=0.152,
                            right=0.974,
                            hspace=0.2,
                            wspace=0.2)



        f = plt.gcf()
        f.savefig(self.folder_path + "/path_comparison.pdf" , pad_inches=0)

    def plot_base_trajectory(self):
        axes= None

        cm_index = [0.8, 0.6, 0.4, 0.2]
        line_width = [2.0, 2.0, 2.0, 2.0, 2.0]
        colors = sns.color_palette("ch:s=.35,rot=-.35", n_colors=3)
        cm = LinearSegmentedColormap.from_list(
            "my_list", colors, N=50)

        # plot base path
        for id, p in enumerate(self.plotters):
            if p.data["name"][:8] == "Baseline":
                sns_color= sns.color_palette("Paired")
                color = SCARLET
            else:
                color = cm(cm_index[id])

            axes = p.plot_base_state_separate(axes, id, linewidth=line_width[id], color=color)

        axes = p.plot_base_ref_separate(axes, id, legend="", linewidth=line_width[id], color=color)

        # axes.set_aspect("equal")
        plt.legend(loc="lower left")
        plt.grid('on')

        # plt.subplots_adjust(top=0.99,
        #                     bottom=0.166,
        #                     left=0.083,
        #                     right=0.974,
        #                     hspace=0.15,
        #                     wspace=0.16)

        f = plt.gcf()
        f.savefig(self.folder_path + "/trajectory_comparison.pdf" , pad_inches=0)

    def plot_sdf_distance(self):

        f, axes = plt.subplots(1, 1, sharex=True, figsize=(3.5, 2.0))
        t_perceptive = self.plotters[0].data['raw']['xs']['ts'][self.t_idx_perceptive_raw]
        axes.axvspan(t_perceptive, 8, color="green", alpha=0.1)  # alpha for transparency
        axes.axvline(t_perceptive, color='darkgreen', linestyle='--')  # Optional visual for x-value

        arrow_length = 2.5
        y = 0.8
        axes.annotate(
            '', 
            xy=(t_perceptive + arrow_length, y), 
            xytext=(t_perceptive+arrow_length*0.05, y),
            arrowprops=dict(arrowstyle='->', linewidth=2)
        )

        # Plot left-pointing arrow
        axes.annotate(
            '', 
            xy=(t_perceptive - arrow_length, y), 
            xytext=(t_perceptive-arrow_length*0.05, y),
            arrowprops=dict(arrowstyle='->', linewidth=2)
        )

        # Add text annotation at the center
        axes.text(
            t_perceptive+arrow_length/2, y, "Perceptive",
            ha='center', va='center', 
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
        )
        axes.text(
            t_perceptive-arrow_length/2, y, "Blinded",
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
        )
        cm_index = [0.8, 0.6, 0.4, 0.2]
        line_width = [2.0, 2.0, 2.0, 2.0, 2.0]
        colors = sns.color_palette("ch:s=.35,rot=-.35", n_colors=3)
        cm = LinearSegmentedColormap.from_list(
            "my_list", colors, N=50)

        # plot base path
        for id, p in enumerate(self.plotters):
            if p.data["name"][:8] == "Baseline":
                sns_color= sns.color_palette("Paired")
                color = SCARLET
            else:
                color = cm(cm_index[id])

            axes = p.plot_sdf_collision(axes, id, linewidth=line_width[id], color=color)

        plt.axhline(y=0., color=SCARLET, linestyle='-', linewidth=2, label="Safety Boundary")

        # axes.set_aspect("equal")
        plt.legend(loc="lower left")
        plt.grid('on')
        axes.set_xlim([0, 6.8])
        axes.set_title("Signed Distance (Robot Wholebody to Obstalces)")


        plt.subplots_adjust(top=0.87,
                            bottom=0.181,
                            left=0.148,
                            right=0.967,
                            hspace=0.2,
                            wspace=0.2)

        f = plt.gcf()
        f.savefig(self.folder_path + "/sdf_distance.pdf" , pad_inches=0)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", required=True, help="Path to data folder.")
    parser.add_argument("--htmpc_vs_stmpc", action="store_true", help="export figures to file")
    parser.add_argument("--base_path", action="store_true", help="export figures to file")
    parser.add_argument("--sdf_distance", action="store_true", help="export figures to file")
    parser.add_argument("--show", action="store_true", help="export figures to file")



    args = parser.parse_args()

    plotter = SemiStaticBenchmarkPlotter(args)
    if args.htmpc_vs_stmpc:
        plotter.plot_base_path_htmpc_vs_stmpc()
    if args.base_path:
        plotter.plot_base_path()
    # plotter.plot_base_trajectory()
    if args.sdf_distance:
        plotter.plot_sdf_distance()
    if args.show:
        plt.show()


