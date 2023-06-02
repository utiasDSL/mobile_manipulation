# Minor modification based on original implementation by Adam Heins
# ref: https://github.com/utiasDSL/dsl__projects__tray_balance/blob/master/upright_core/src/upright_core/logging.py

import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import rosbag
import os
from spatialmath.base import rotz, r2q

from mmseq_utils.parsing import parse_path, load_config
from mmseq_control.robot import MobileManipulator3D
from matplotlib.backends.backend_pdf import PdfPages
from mobile_manipulation_central import ros_utils

VICON_TOOL_NAME = "ThingContainer"

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, dpi=dpi, format='pdf')
    pp.close()

class DataLogger:
    """Log data for later saving and viewing."""

    def __init__(self, config):
        self.directory = Path(parse_path(config["logging"]["log_dir"]))
        # self.timestep = config["logging"]["timestep"]
        # self.last_log_time = -np.infty

        self.config = config
        self.data = {}

    # # TODO it may bite me that this is stateful
    # def ready(self, t):
    #     if t >= self.last_log_time + self.timestep:
    #         self.last_log_time = t
    #         return True
    #     return False

    def add(self, key, value):
        """Add a single value named `key`."""
        if key in self.data:
            raise ValueError(f"Key {key} already in the data log.")
        self.data[key] = value

    def append(self, key, value):
        """Append a values to the list named `key`."""
        # copy to an array (also copies if value is already an array, which is
        # what we want)
        a = np.array(value)

        # append to list or start a new list if this is the first value under
        # `key`
        if key in self.data:
            if a.shape != self.data[key][-1].shape:
                raise ValueError("Data must all be the same shape.")
            self.data[key].append(a)
        else:
            self.data[key] = [a]

    def save(self, timestamp, name="data"):
        """Save the data and configuration to a timestamped directory."""
        dir_name = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        dir_name = name + "_" + dir_name
        dir_path = self.directory / dir_name
        dir_path.mkdir(parents=True)

        data_path = dir_path / "data.npz"
        config_path = dir_path / "config.yaml"

        self.data["dir_path"] = str(dir_path)

        # save the recorded data
        np.savez_compressed(data_path, **self.data)

        # save the configuration used for this run
        with open(config_path, "w") as f:
            yaml.safe_dump(self.config, stream=f, default_flow_style=False)

        print(f"Saved data to {dir_path}.")


class DataPlotter:
    def __init__(self, data):
        self.data = data
        self.data["name"] = self.data.get('name', 'data')

    @classmethod
    def from_logger(cls, logger):
        # convert logger data to numpy format
        data = {}
        for key, value in logger.data.items():
            data[key] = np.array(value)
        return cls(data)

    @classmethod
    def from_npz(cls, npz_file_path):
        data = dict(np.load(npz_file_path))
        if "name" not in data:
            path_split = npz_file_path.split("/")
            folder_name = path_split[-2]
            data_name = folder_name.split("_")[:-2]
            data_name = "_".join(data_name)
            data["name"] = data_name
        return cls(data)

    @classmethod
    def from_ROS_results(cls, folder_path):
        data_decoupled = {}
        for filename in os.listdir(folder_path):
            d = os.path.join(folder_path, filename)
            key = filename.split("_")[0]
            if os.path.isdir(d):
                path_to_npz = os.path.join(d, "data.npz")
                data_decoupled[key] = dict(np.load(path_to_npz))

        data = data_decoupled["control"]

        t = data["ts"]
        t_sim = data_decoupled["sim"]["ts"]
        for key, value in data_decoupled["sim"].items():
            if key in ["ts", "sim_timestep", "nq", "nv", "nx", "nu", "duration", "dir_path", "cmd_vels"]:
                continue
            else:
                value = np.array(value)
                f_interp = interp1d(t_sim, value, axis=0, fill_value="extrapolate")
                data[key] = f_interp(t)
        data["ts"] -= data["ts"][0]
        data["name"] = folder_path.split("/")[-1]
        return cls(data)

    def plot_ee_position(self, axes=None, index=0, legend=None):
        ts = self.data["ts"]
        r_ew_w_ds = self.data.get("r_ew_w_ds", [])
        r_ew_ws = self.data.get("r_ew_ws", [])

        if len(r_ew_w_ds) == 0 and len(r_ew_ws) == 0:
            return

        if axes is None:
            axes = []
            f, axes = plt.subplots(1, 1, sharex=True)

        if legend is None:
            legend = self.data["name"]

        if len(r_ew_w_ds) > 0:
            axes.plot(ts, r_ew_w_ds[:, 0], label=legend + "$x_d$", color="r", linestyle="--")
            axes.plot(ts, r_ew_w_ds[:, 1], label=legend + "$y_d$", color="g", linestyle="--")
            axes.plot(ts, r_ew_w_ds[:, 2], label=legend + "$z_d$", color="b", linestyle="--")
        if len(r_ew_ws) > 0:
            axes.plot(ts, r_ew_ws[:, 0], label=legend + "$x$", color="r")
            axes.plot(ts, r_ew_ws[:, 1], label=legend + "$y$", color="g")
            axes.plot(ts, r_ew_ws[:, 2], label=legend + "$z$", color="b")
        axes.grid()
        axes.legend()
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Position (m)")
        axes.set_title("End effector position")

        return axes


    def plot_base_position(self, axes=None, index=0, legend=None):
        ts = self.data["ts"]
        r_ew_w_ds = self.data.get("r_bw_w_ds", [])
        r_ew_ws = self.data.get("r_bw_ws",[])
        if len(r_ew_w_ds) == 0 and len(r_ew_ws) == 0:
            return

        if axes is None:
            axes = []
            f, axes = plt.subplots(1, 1, sharex=True)

        if legend is None:
            legend = self.data["name"]
        if len(r_ew_w_ds) > 0:
            axes.plot(ts, r_ew_w_ds[:, 0], label=legend + "$x_d$", color="r", linestyle="--")
            axes.plot(ts, r_ew_w_ds[:, 1], label=legend + "$y_d$", color="g", linestyle="--")
        if len(r_ew_ws) > 0:
            axes.plot(ts, r_ew_ws[:, 0], label=legend + "$x$", color="r")
            axes.plot(ts, r_ew_ws[:, 1], label=legend + "$y$", color="g")
        axes.grid()
        axes.legend()
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Position (m)")
        axes.set_title("Base position")

        return axes

    def plot_ee_orientation(self, axes=None, index=0, legend=None):
        ts = self.data["ts"]
        Q_we_ds = self.data["Q_we_ds"]
        Q_wes = self.data["Q_wes"]

        if axes is None:
            axes = []
            f, axes = plt.subplots(1, 1, sharex=True)

        if legend is None:
            legend = self.data["name"]

        axes.plot(ts, Q_we_ds[:, 0], label=legend + "$Q_{d,x}$", color="r", linestyle="--")
        axes.plot(ts, Q_we_ds[:, 1], label=legend + "$Q_{d,y}$", color="g", linestyle="--")
        axes.plot(ts, Q_we_ds[:, 2], label=legend + "$Q_{d,z}$", color="b", linestyle="--")
        axes.plot(ts, Q_wes[:, 0], label=legend + "$Q_{x}$", color="r")
        axes.plot(ts, Q_wes[:, 1], label=legend + "$Q_{y}$", color="g")
        axes.plot(ts, Q_wes[:, 2], label=legend + "$Q_{z}$", color="b")
        axes.grid()
        axes.legend()
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Orientation")
        axes.set_title("End effector orientation")

    def plot_ee_velocity(self, axes=None, index=0, legend=None):
        ts = self.data["ts"]
        v_ew_ws = self.data["v_ew_ws"]
        ω_ew_ws = self.data["ω_ew_ws"]

        if axes is None:
            axes = []
            f, axes = plt.subplots(1, 1, sharex=True)
        if legend is None:
            legend = self.data["name"]

        axes.plot(ts, v_ew_ws[:, 0], label=legend + "$v_x$")
        axes.plot(ts, v_ew_ws[:, 1], label=legend + "$v_y$")
        axes.plot(ts, v_ew_ws[:, 2], label=legend + "$v_z$")
        axes.plot(ts, ω_ew_ws[:, 0], label=legend + "$ω_x$")
        axes.plot(ts, ω_ew_ws[:, 1], label=legend + "$ω_y$")
        axes.plot(ts, ω_ew_ws[:, 2], label=legend + "$ω_z$")
        axes.grid()
        axes.legend()
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Velocity")
        axes.set_title("End effector velocity")

        return axes

    def plot_tracking_err(self, axes=None, index=0, legend=None):
        plot_base_err = False
        plot_ee_err = False

        ts = self.data["ts"]
        r_bw_w_ds = self.data.get("r_bw_w_ds", [])
        r_bw_ws = self.data.get("r_bw_ws", [])
        if len(r_bw_ws) > 0 and len(r_bw_w_ds) > 0:
            plot_base_err = True
            err_base = np.linalg.norm(r_bw_ws - r_bw_w_ds, axis=1)
            rms_base = np.mean(err_base * err_base) ** 0.5


        r_ew_w_ds = self.data.get("r_ew_w_ds", [])
        r_ew_ws = self.data.get("r_ew_ws", [])
        if len(r_ew_ws) > 0 and len(r_ew_w_ds) > 0:
            plot_ee_err = True
            err_ee = np.linalg.norm(r_ew_ws - r_ew_w_ds, axis=1)
            rms_ee = np.mean(err_ee * err_ee) ** 0.5

        if axes is None:
            axes = []
            f, axes = plt.subplots(1, 1, sharex=True)

        if legend is None:
            legend = self.data["name"]

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        if plot_base_err:
            axes.plot(ts, err_base, label=legend+" $err_{base}, rms_{base} = $" + str(rms_base), linestyle="--", color=colors[index])
        if plot_ee_err:
            axes.plot(ts, err_ee, label=legend+" $err_{ee}, rms_{ee} = $" + str(rms_ee),  linestyle="-", color=colors[index])
        axes.grid()
        axes.legend()
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Err (m)")
        axes.set_title("Tracking Error vs Time")

        return axes

    def plot_cmds(self, axes=None, index=0, legend=None):
        ts = self.data["ts"]
        cmd_vels = self.data["cmd_vels"]
        cmd_accs = self.data.get("cmd_accs")
        nq = int(self.data["nq"])
        nv = int(self.data["nv"])

        if axes is None:
            axes = []
            for i in range(2):
                f, ax = plt.subplots(nv, 1, sharex=True)
                axes.append(ax)
        if legend is None:
            legend = self.data["name"]

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        ax = axes[0]
        for i in range(nv):
            ax[i].plot(
                ts,
                cmd_vels[:, i],
                '-x',
                label=legend + f"$v_{{cmd_{i + 1}}}$" + f"max = " + str(max(cmd_vels[:, i])),
                linestyle="--",
                color=colors[index],
            )

            ax[i].grid()
            ax[i].legend()
        ax[-1].set_xlabel("Time (s)")
        ax[0].set_title("Commanded joint velocity (rad/s)")

        if cmd_accs is not None:
            ax = axes[1]
            for i in range(nv):
                ax[i].plot(
                    ts,
                    cmd_accs[:, i],
                    '-x',
                    label=legend + f"$a_{{cmd_{i + 1}}}$" + f"max = " + str(max(cmd_accs[:, i])),
                    linestyle="--",
                    color=colors[index],
                )

                ax[i].grid()
                ax[i].legend()
            ax[-1].set_xlabel("Time (s)")
            ax[0].set_title("Commanded joint acceleration (rad/s^2)")

        return axes

    def plot_du(self, axes=None, legend=None):
        ts = self.data["ts"]
        cmd_accs = self.data.get("cmd_accs")
        if cmd_accs is not None:
            cmd_dacc = cmd_accs[1:, :] - cmd_accs[:-1, :]
            nq = int(self.data["nq"])
            nv = int(self.data["nv"])

            if axes is None:
                axes = []
                f, axes = plt.subplots(nv, 1, sharex=True)
            if legend is None:
                legend = self.data["name"]

            prop_cycle = plt.rcParams["axes.prop_cycle"]
            colors = prop_cycle.by_key()["color"]

            for i in range(nv):
                axes[i].plot(
                    ts[:-1],
                    cmd_dacc[:, i],
                    '-x',
                    label=legend + f"$v_{{cmd_{i + 1}}}$",
                    color=colors[i],
                )

                axes[i].grid()
                axes[i].legend()
            axes[-1].set_xlabel("Time (s)")
            axes[0].set_title("Commanded joint acceleration time difference (rad/s^3)")

        return axes

    def plot_cmd_vs_real_vel(self, axes=None):
        ts = self.data["ts"]
        xs = self.data["xs"]
        cmd_vels = self.data["cmd_vels"]
        nq = int(self.data["nq"])
        nv = int(self.data["nv"])

        if axes is None:
            axes = []
            f, axes = plt.subplots(nv, 1, sharex=True)

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        for i in range(nv):
            axes[i].plot(ts, xs[:, nq + i], label=f"$v_{i+1}$", color=colors[i])
            axes[i].plot(
                ts,
                cmd_vels[:, i],
                label=f"$v_{{cmd_{i + 1}}}$",
                linestyle="--",
                color=colors[i],
            )

            axes[i].grid()
            axes[i].legend()
        axes[-1].set_xlabel("Time (s)")
        axes[0].set_title("Actual and commanded joint velocity (rad/s)")

        return axes

    def _plot_line_value_vs_time(self, key, legend_prefix, indices=None):
        ts = self.data["ts"]
        ys = self.data[key]

        if indices is not None:
            min_idx = np.min(indices)
            for idx in indices:
                # triple {{{ required to to f-string substitution and leave a
                # literal { for latex
                plt.plot(ts, ys[:, idx], label=f"${legend_prefix}_{{{idx+1-min_idx}}}$")
        elif len(ys.shape) > 1:
            for idx in range(ys.shape[1]):
                plt.plot(ts, ys[:, idx], label=f"${legend_prefix}_{{{idx+1}}}$")
        else:
            plt.plot(ts, ys)

    # TODO rewrite more functions in terms of this (probably eliminate a lot
    # of them)
    def plot_value_vs_time(self, key, indices=None, legend_prefix=None, ylabel=None, title=None):
        """Plot the value stored in `key` vs. time."""
        if key not in self.data:
            print(f"Key {key} not found, skipping plot.")
            return

        if legend_prefix is None:
            legend_prefix = key
        if ylabel is None:
            ylabel = key
        if title is None:
            title = f"{key} vs time"

        fig = plt.figure()

        self._plot_line_value_vs_time(key, legend_prefix, indices=indices)

        plt.grid()
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.legend(loc="upper right")
        plt.title(title)

        ax = plt.gca()
        return ax

    def plot_state(self):
        self.plot_value_vs_time(
            "xs",
            indices=range(self.data["nq"]),
            legend_prefix="q",
            ylabel="Joint Position (rad)",
            title="Joint Positions vs. Time",
        )
        self.plot_value_vs_time(
            "xs",
            indices=range(self.data["nq"], self.data["nq"] + self.data["nv"]),
            legend_prefix="v",
            ylabel="Joint Velocity (rad/s)",
            title="Joint Velocities vs. Time",
        )
        # self.plot_value_vs_time(
        #     "xs",
        #     indices=range(
        #         self.data["nq"] + self.data["nv"], self.data["nq"] + 2 * self.data["nv"]
        #     ),
        #     legend_prefix="a",
        #     ylabel="Joint Acceleration (rad/s^2)",
        #     title="Joint Accelerations vs. Time",
        # )
        #
        # # plot the obstacle position if available
        # if self.data["xs"].shape[1] > self.data["nx"]:
        #     self.plot_value_vs_time(
        #         "xs",
        #         indices=range(self.data["nx"], self.data["nx"] + 3),
        #         legend_prefix="r",
        #         ylabel="Obstacle position (m)",
        #         title="Obstacle Position",
        #     )
    def plot_run_time(self, axes=None, index=0, block=True, legend=None):
        # Time x HT-Iter x Task x ST-ITer+1
        t_sim = self.data["ts"]
        run_time = self.data.get("controller_run_time")
        if run_time is None:
            print("Ignore run time")
            return

        if legend is None:
            legend = self.data["name"]

        if axes is None:
            axes = []
            f, ax = plt.subplots(1, 1)
            axes.append(ax)

        ax = axes[0]
        ax.plot(t_sim, run_time*1000, label=legend, linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("run time (ms)")
        ax.legend(fontsize=20)
        ax.set_title("Controller Run Time")

        return axes


    def plot_cost_htmpc(self, axes=None, index=0, block=True, legend=None):
        '''

        :param log:
        :param robot:
        :param axes: array of 2 subplots axes
        :param index:
        :param block:
        :return:
        '''
        # Time x HT-Iter x Task x ST-ITer+1
        t_sim = self.data["ts"]
        cost = self.data.get("mpc_cost_iters")
        cost_final = self.data.get("mpc_cost_finals")
        if cost is None or cost_final is None:
            print("Ignore cost_htmpc")
            return

        N, HT_iter_num, task_num, ST_iter_num = cost.shape

        if legend is None:
            legend = self.data["name"]

        if axes is None:
            axes = []
            for k in range(2):
                f, ax = plt.subplots(task_num, 1, sharex=True)
                axes.append(ax)

        # Fig 1: Cost for each task over time and iterations
        if task_num > 1:
            ax = axes[0]
        else:
            ax = [axes[0]]

        xticks = np.arange(N) * (HT_iter_num * ST_iter_num)
        xlables = [str(t) for t in t_sim]
        print(cost.shape)
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        for l in range(task_num):
            ax[l].plot(cost[:, :, l, :].flatten(), "x-", label=legend + " iter", linewidth=3, markersize=10)
            ax[l].plot(xticks + HT_iter_num * ST_iter_num - 1, cost_final[:, l], "x-",
                       label=legend + " final", linewidth=2, markersize=8)
            ax[l].set_title("Task" + str(l) + " Cost", fontsize=20)
            ax[l].set_xticks(xticks)
        ax[-1].set_xticklabels(xlables)
        ax[0].legend(fontsize=20)
        # plt.show(block=False)

        # Fig 2: Cost for each task over time
        if task_num > 1:
            ax = axes[1]
        else:
            ax = [axes[1]]

        for l in range(task_num):
            ax[l].plot(t_sim, cost_final[:, l], ".-", label=legend, linewidth=3, markersize=10)
            ax[l].set_title("Task " + str(l) + " Final Cost", fontsize=20)

        ax[0].legend(fontsize=20)
        # plt.show(block=block)
        return axes

    def plot_solver_status_htmpc(self, axes=None, index=0, block=True, legend=None):
        # Time x HT-Iter x Task x ST-ITer
        status = self.data.get("mpc_solver_statuss")
        step_size = self.data.get("mpc_step_sizes")
        if status is None or step_size is None:
            print("Ignore solver status")
            return

        N, HT_iter_num, task_num, ST_iter_num = status.shape
        t_sim = self.data["ts"]
        if legend is None:
            legend = self.data["name"]

        if axes is None:
            f, axes = plt.subplots(task_num, 1, sharex=True)
        if task_num == 1:
            axes = [axes]
        xticks = np.arange(N) * (HT_iter_num * ST_iter_num)
        xlables = [str(t) for t in t_sim]
        for l in range(task_num):
            axes[l].plot(status[:, :, l, :].flatten(), "x-", label=legend + " solver status", linewidth=2, markersize=8)
            axes[l].plot(step_size[:, :, l, :].flatten(), "x-", label=legend + " step size", linewidth=2, markersize=8)
            axes[l].set_title("Task" + str(l))
            axes[l].set_xticks(xticks)
            axes[l].set_xticklabels(xlables)
        axes[0].legend()
        # plt.show(block=block)

        return axes

    def plot_task_violation(self, axes=None, index=0, legend=None):
        task_names = self.data.get("task_names")
        if task_names is None:
            return
        else:
            task_names = task_names[0]
        ws = []
        for name in task_names:
            ws.append(self.data.get("task_violations_" + name))

        task_num = len(task_names)
        if axes is None:
            f, axes = plt.subplots(task_num, 1, sharex=True)
        else:
            if len(axes) != task_num:
                raise ValueError("Given axes number ({}) does not match task number ({}).".format(len(axes), task_num))

        if legend is None:
            legend = self.data['name']

        t_sim = self.data["ts"]
        N = len(t_sim)
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        for tid in range(task_num):
            w = ws[tid]

            if tid == 0:
                saturation_num = np.sum(w > 1e-3, axis=1)
                axes[tid].plot(t_sim, saturation_num, color=colors[index], label=legend)
                axes[tid].set_ylabel("# Inequalities Saturated \n out of {}".format(w.shape[1]))
            else:
                axes[tid].plot(t_sim, np.linalg.norm(w, axis=1), color=colors[index], label=legend)
                axes[tid].set_ylabel("Task Violation")

            axes[tid].set_title("Task " + str(tid) + " " + task_names[tid])

        return axes

    def show(self):
        plt.show()

    def plot_all(self):
        self.data["name"] = ""
        self.plot_ee_position()
        self.plot_base_position()
        self.plot_tracking_err()
        self.plot_cmd_vs_real_vel()
        self.plot_state()

        self.plot_cost_htmpc()
        self.plot_solver_status_htmpc()
        self.plot_run_time()

        figs = [plt.figure(n) for n in plt.get_fignums()]
        print(self.data["dir_path"])
        multipage(Path(str(self.data["dir_path"])) / "data.pdf", figs)

    def plot_mpc(self):
        self.plot_cost_htmpc()
        self.plot_solver_status_htmpc()
        self.plot_run_time()

    def plot_robot(self):
        self.plot_cmd_vs_real_vel()
        self.plot_state()
        self.plot_cmds()
        self.plot_du()

    def plot_tracking(self):
        self.plot_ee_position()
        self.plot_base_position()
        self.plot_tracking_err()
        self.plot_cmd_vs_real_vel()

class ROSBagPlotter:
    def __init__(self, bag_file, config_file="/home/tracy/Projects/mm_catkin_ws/src/mm_sequential_tasks/mmseq_run/config/robot/thing.yaml"):
        self.data = {"ur10": {}, "ridgeback": {}, "mpc": {}, "vicon": {}, "model":{}}
        self.bag = rosbag.Bag(bag_file)
        self.config = load_config(config_file)
        self.robot = MobileManipulator3D(self.config["controller"])

        self.parse_joint_states(self.bag)
        self.parse_cmd_vels(self.bag)
        self.parse_mpc_tracking_pt(self.bag)
        self.parse_vicon_msgs(self.bag)
        self.compute_values_from_robot_model()

        self._set_zero_time()

    def parse_joint_states(self, bag):

        ur10_msgs = [msg for _, msg, _ in bag.read_messages("/ur10/joint_states")]
        ridgeback_msgs = [msg for _, msg, _ in bag.read_messages("/ridgeback/joint_states")]

        tas, qas, vas = ros_utils.parse_ur10_joint_state_msgs(ur10_msgs, False)
        tbs, qbs, vbs = ros_utils.parse_ridgeback_joint_state_msgs(ridgeback_msgs, False)
        self.data["ur10"]["joint_states"] = {"ts": tas, "qs": qas, "vs": vas}           # 125hz
        self.data["ridgeback"]["joint_states"] = {"ts": tbs, "qs": qbs, "vs": vbs}      # 50hz

        # Reconstruct body-frame ridgeback velocity
        vb_bs = [rotz(qbs[i, 2]).T @  vbs[i, :] for i in range(len(tbs))]
        self.data["ridgeback"]["joint_states"]["vbs"] = np.array(vb_bs)

        fqa_interp = interp1d(tas, qas, axis=0, fill_value="extrapolate")
        fva_interp = interp1d(tas, vas, axis=0, fill_value="extrapolate")
        qas_interp = fqa_interp(tbs)
        vas_interp = fva_interp(tbs)
        self.data["ur10"]["joint_states_interpolated"] = {"ts": tbs, "qs": qas_interp, "vs": vas_interp}

    def parse_cmd_vels(self, bag):
        # cmd_vel messages do not have header.
        # we use the time received by the bag recording node for plotting

        # UR 10
        ur10_ts = np.array([t.to_sec() for _, _, t in bag.read_messages("/ur10/cmd_vel")])
        ur10_cmd_msgs = [msg for _, msg, _ in bag.read_messages("/ur10/cmd_vel")]
        ur10_cmd_vels = ros_utils.parse_ur10_cmd_vel_msgs(ur10_cmd_msgs)

        # ridgeback
        ridgeback_ts = np.array([t.to_sec() for _, _, t in bag.read_messages("/ridgeback/cmd_vel")])
        ridgeback_cmd_msgs = [msg for _, msg, _ in bag.read_messages("/ridgeback/cmd_vel")]
        # ridgeback cmd_vels are in body frame
        ridgeback_cmd_vels = ros_utils.parse_ridgeback_cmd_vel_msgs(ridgeback_cmd_msgs)

        self.data["ur10"]["cmd_vels"] = {"ts": ur10_ts, "vcs": ur10_cmd_vels}
        self.data["ridgeback"]["cmd_vels"] = {"ts": ridgeback_ts, "vc_bs": ridgeback_cmd_vels}

    def parse_mpc_tracking_pt(self, bag):
        tracking_pt_msgs = [msg for _, msg, _ in bag.read_messages("/mpc_tracking_pt")]
        t_rs, pose_rs = ros_utils.parse_multidofjointtrajectory_msg(tracking_pt_msgs, False)
        self.data["mpc"]["tracking_pt"] = {"ts": t_rs,
                                           "rees": pose_rs.get("EE", []),
                                           "rbs": pose_rs.get("base", [])}

    def parse_vicon_msgs(self, bag):
        topics = bag.get_type_and_topic_info()[1].keys()
        vicon_topics = [topic for topic in topics if topic[1:6] == "vicon"]
        print(topics)
        for topic in vicon_topics:
            name = topic.split("/")[-1]
            if name == 'markers':
                continue
            msgs = [msg for _, msg, _ in bag.read_messages(topic)]
            if name == "ThingBase":
                ts, qs = ros_utils.parse_ridgeback_vicon_msgs(msgs)         # q is a 3-element vector x, y, theta
                self.data["vicon"][name] = {"ts": ts, "pos": qs[:, :2], "orn": qs[:, 2]}
            else:
                ts, qs = ros_utils.parse_transform_stamped_msgs(msgs, False)    # q is a 7-element vector x, y, z for position followed by a unit quaternion
                self.data["vicon"][name] = {"ts": ts, "pos": qs[:, :3], "orn": qs[:, 3:]}

    def compute_values_from_robot_model(self):
        f_base = self.robot.kinSymMdls[self.robot.base_link_name]
        f_ee = self.robot.kinSymMdls[self.robot.tool_link_name]

        r_base_s = []
        yaw_base_s = []
        r_ee_s = []
        quat_ee_s = []


        for i in range(len(self.data["ur10"]["joint_states_interpolated"]["ts"])):
            qa = self.data["ur10"]["joint_states_interpolated"]["qs"][i]
            qb = self.data["ridgeback"]["joint_states"]["qs"][i]
            q = np.hstack((qb, qa))
            r_b, theta_b = f_base(q)
            r_ee, rot_ee = f_ee(q)
            quat_ee = r2q(np.array(rot_ee), order="xyzs")

            r_base_s.append(r_b)
            yaw_base_s.append(theta_b)

            r_ee_s.append(r_ee)
            quat_ee_s.append(quat_ee)

        self.data["model"]["EE"] = {"ts": self.data["ur10"]["joint_states_interpolated"]["ts"].copy(),
                                    "pos": np.array(r_ee_s),
                                    "orn": np.array(quat_ee_s)}

        self.data["model"]["base"] = {"ts": self.data["ur10"]["joint_states_interpolated"]["ts"].copy(),
                                      "pos": np.array(r_base_s),
                                      "orn": np.array(yaw_base_s).flatten()}

    def _set_zero_time(self):
        if len(self.data["ridgeback"]["cmd_vels"]["ts"]) > 0:
            t0 = self.data["ridgeback"]["cmd_vels"]["ts"][0]
        else:
            t0 = 0

        for k1 in self.data.keys():
            if type(self.data[k1]) is dict:
                for k2 in self.data[k1].keys():
                    self.data[k1][k2]["ts"] -= t0
            else:
                self.data[k1]["ts"] -= t0

    def plot_joint_states(self, axes=None, legend=""):
        if axes is None:
            axes = []
            for i in range(4):
                f = plt.figure()
                axes.append(f.gca())

        ax = axes[0]
        ax.plot(self.data["ridgeback"]["joint_states"]["ts"],
                self.data["ridgeback"]["joint_states"]["qs"], '-', label=["x", "y", "θ"])
        ax.set_title("Ridgeback Joint Positions")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Joint position")
        ax.legend()
        ax.grid()

        ax = axes[1]
        ax.plot(self.data["ridgeback"]["joint_states"]["ts"],
                self.data["ridgeback"]["joint_states"]["vs"], '-', label=[r"$v_x$", r"$v_y$", r"$\omega$"])
        ax.set_title("Ridgeback Twist (World frame)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Twist")
        ax.legend()
        ax.grid()

        ax = axes[2]
        ax.plot(self.data["ur10"]["joint_states"]["ts"],
                self.data["ur10"]["joint_states"]["qs"], '-', label=[r"$\theta_{}$".format(i+1) for i in range(6)])
        ax.set_title("UR10 Joint Positions")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Joint position")
        ax.legend()
        ax.grid()

        ax = axes[3]
        ax.plot(self.data["ur10"]["joint_states"]["ts"],
                self.data["ur10"]["joint_states"]["vs"], '-', label=[r"$\dot\theta_{}$".format(i+1) for i in range(6)])
        ax.set_title("UR10 Joint Velocities")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Joint Velocity")
        ax.legend()
        ax.grid()

    def plot_joint_vel_tracking(self, axes=None, legend=""):
        if axes is None:
            f1, ax1 = plt.subplots(6, 1, sharex=True)
            f2, ax2 = plt.subplots(3, 1, sharex=True)
            axes = [ax1, ax2]

        ax = axes[0]
        for i in range(6):
            ax[i].plot(self.data["ur10"]["joint_states"]["ts"],
                    self.data["ur10"]["joint_states"]["vs"][:, i], '-',
                    label=r"$\dot\theta_{}$".format(i + 1))
            ax[i].plot(self.data["ur10"]["cmd_vels"]["ts"],
                    self.data["ur10"]["cmd_vels"]["vcs"][:, i], '-',
                    label=r"${\dot\theta_d}" + "_{}$".format(i + 1))
            ax[i].legend()
            ax[i].grid()
        ax[0].set_title("UR10 Joint Velocity Tracking")
        ax[-1].set_xlabel("Time (s)")

        ax = axes[1]
        labels_v = [r"$v_x$", r"$v_y$", r"$\omega$"]
        labels_vc = [r"$v_{c,x}$", r"$v_{c, y}$", r"$\omega_c$"]
        for i in range(3):
            ax[i].plot(self.data["ridgeback"]["joint_states"]["ts"],
                   self.data["ridgeback"]["joint_states"]["vbs"][:, i], '-',
                   label=labels_v[i])

            ax[i].plot(self.data["ridgeback"]["cmd_vels"]["ts"],
                   self.data["ridgeback"]["cmd_vels"]["vc_bs"][:, i], '--',
                   label=labels_vc[i])
        for i in range(3):
            ax[i].legend()
        ax[0].set_title("Ridgeback Velocity Tracking (Body Frame)")
        ax[2].set_xlabel("Time (s)")


    def plot_tracking(self, axes=None, subscript=""):
        t_ref = self.data["mpc"]["tracking_pt"]["ts"]
        rees = self.data["mpc"]["tracking_pt"].get("rees", [])
        rbs  = self.data["mpc"]["tracking_pt"].get("rbs", [])

        plot_ee_tracking = True if len(rees) > 0 else False
        plot_base_tracking = True if len(rbs) > 0 else False

        if axes is None:
            axes = []

            for i in range(int(plot_ee_tracking + plot_base_tracking)):
                fi = plt.figure()
                axes.append(fi.gca())

        curr_axes_indx = 0
        if plot_ee_tracking:
            ax = axes[curr_axes_indx]
            curr_axes_indx += 1
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            colors = prop_cycle.by_key()["color"]
            labels = ['x', 'y', 'z']
            for i in range(3):
                ax.plot(t_ref, rees[:, i], '--', label=labels[i] + "d" +subscript, color=colors[i])
                ax.plot(self.data["vicon"][VICON_TOOL_NAME]["ts"],
                    self.data["vicon"][VICON_TOOL_NAME]["qs"][:, i], '-',
                    label=labels[i] + subscript, color=colors[i])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Positions")
            ax.set_title("EE Tracking Performance")
            ax.legend()
            ax.grid()

        if plot_base_tracking:
            ax = axes[curr_axes_indx]
            curr_axes_indx += 1
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            colors = prop_cycle.by_key()["color"]
            labels = ['x', 'y']
            for i in range(2):
                ax.plot(t_ref, rbs[:, i], '--', label=labels[i] + "d" + subscript, color=colors[i])
                ax.plot(self.data["ridgeback"]["joint_states"]["ts"],
                    self.data["ridgeback"]["joint_states"]["qs"][:, i], '-',
                    label=labels[i] + subscript, color=colors[i])

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Positions")
            ax.set_title("Base Tracking Performance")
            ax.legend()
            ax.grid()

        return axes

    def plot_model_vs_groundtruth(self):
        f1 = plt.figure()
        pos_label = ['x', 'y', 'z']
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        for i in range(3):
            plt.plot(self.data["model"]["EE"]["ts"], self.data["model"]["EE"]["pos"][:, i],
                     label=pos_label[i] + "_model", color=colors[i])
            plt.plot(self.data["vicon"][VICON_TOOL_NAME]["ts"], self.data["vicon"][VICON_TOOL_NAME]["pos"][:, i], '--',
                     label=pos_label[i] + "_meas", color=colors[i])

        plt.title("End Effector Position")
        plt.legend()

        f2 = plt.figure()
        orn_label = ['x', 'y', 'z', 'w']
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        for i in range(4):
            plt.plot(self.data["model"]["EE"]["ts"], self.data["model"]["EE"]["orn"][:, i],
                     label=orn_label[i] + "_model", color=colors[i])
            plt.plot(self.data["vicon"][VICON_TOOL_NAME]["ts"], self.data["vicon"][VICON_TOOL_NAME]["orn"][:, i], '--',
                     label=orn_label[i] + "_meas", color=colors[i])

        plt.title("End Effector Orientation (Quaternion)")
        plt.legend()

        # f3 = plt.figure()
        # label = ['x', 'y', 'Θ']
        # prop_cycle = plt.rcParams["axes.prop_cycle"]
        # colors = prop_cycle.by_key()["color"]
        # for i in range(2):
        #     plt.plot(self.data["model"]["base"]["ts"], self.data["model"]["base"]["pos"][:, i],
        #              label=label[i] + "_model", color=colors[i])
        #     plt.plot(self.data["vicon"]["ThingBase"]["ts"], self.data["vicon"]["ThingBase"]["pos"][:, i], '--',
        #              label=label[i] + "_meas", color=colors[i])
        #
        # plt.plot(self.data["model"]["base"]["ts"], self.data["model"]["base"]["orn"],
        #          label=label[2] + "_model", color=colors[2])
        # plt.plot(self.data["vicon"]["ThingBase"]["ts"], self.data["vicon"]["ThingBase"]["orn"], '--',
        #          label=label[2] + "_meas", color=colors[2])
        #
        # plt.title("Base Pose")
        # plt.legend()

    def plot_show(self):
        plt.show()










