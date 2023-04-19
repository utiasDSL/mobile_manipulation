# Minor modification based on original implementation by Adam Heins
# ref: https://github.com/utiasDSL/dsl__projects__tray_balance/blob/master/upright_core/src/upright_core/logging.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from mmseq_utils.parsing import parse_path
from matplotlib.backends.backend_pdf import PdfPages

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
        ts = self.data["ts"]
        r_bw_w_ds = self.data["r_bw_w_ds"]
        r_bw_ws = self.data["r_bw_ws"]
        err_base = np.linalg.norm(r_bw_ws - r_bw_w_ds, axis=1)

        r_ew_w_ds = self.data["r_ew_w_ds"]
        r_ew_ws = self.data["r_ew_ws"]
        err_ee = np.linalg.norm(r_ew_ws - r_ew_w_ds, axis=1)

        if axes is None:
            axes = []
            f, axes = plt.subplots(1, 1, sharex=True)

        if legend is None:
            legend = self.data["name"]

        axes.plot(ts, err_base, label=legend+"$err_{base}$", linestyle="--")
        axes.plot(ts, err_ee, label=legend+"$err_{ee}$",  linestyle="--")
        axes.grid()
        axes.legend()
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("RMS Err (m)")
        axes.set_title("Tracking Error vs Time")

        return axes

    def plot_cmds(self, axes=None, legend=None):
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
                label=legend + f"$v_{{cmd_{i + 1}}}$",
                linestyle="--",
                color=colors[i],
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
                    label=legend + f"$a_{{cmd_{i + 1}}}$",
                    linestyle="--",
                    color=colors[i],
                )

                ax[i].grid()
                ax[i].legend()
            ax[-1].set_xlabel("Time (s)")
            ax[0].set_title("Commanded joint acceleration (rad/s^2)")

        return axes

    def plot_du(self, axes=None, legend=None):
        ts = self.data["ts"]
        cmd_accs = self.data["cmd_accs"]
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
        # self.plot_run_time()

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

    def plot_tracking(self):
        self.plot_ee_position()
        self.plot_base_position()
        self.plot_tracking_err()
        self.plot_cmd_vs_real_vel()


