"""
Trajectory and path plotting functionality.

This module provides plotting methods for visualizing robot trajectories,
tracking performance, and task execution metrics.
"""

import matplotlib.pyplot as plt
import numpy as np


class TrajectoryPlotterMixin:
    """Mixin class containing trajectory and path plotting methods."""

    def plot_ee_tracking(self):
        """Plot end-effector position tracking."""
        ts = self.data["ts"]
        r_ew_w_ds = self.data.get("r_ew_w_ds", [])
        r_ew_ws = self.data.get("r_ew_ws", [])

        if len(r_ew_w_ds) == 0 and len(r_ew_ws) == 0:
            return

        _, axes = plt.subplots(1, 1, sharex=True)
        legend = self.data["name"]

        if len(r_ew_w_ds) > 0:
            axes.plot(
                ts, r_ew_w_ds[:, 0], label=legend + "$x_d$", color="r", linestyle="--"
            )
            axes.plot(
                ts, r_ew_w_ds[:, 1], label=legend + "$y_d$", color="g", linestyle="--"
            )
            axes.plot(
                ts, r_ew_w_ds[:, 2], label=legend + "$z_d$", color="b", linestyle="--"
            )
        if len(r_ew_ws) > 0:
            axes.plot(ts, r_ew_ws[:, 0], label=legend + "$x$", color="r")
            axes.plot(ts, r_ew_ws[:, 1], label=legend + "$y$", color="g")
            axes.plot(ts, r_ew_ws[:, 2], label=legend + "$z$", color="b")
        axes.grid()
        axes.legend()
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Position (m)")
        axes.set_title("End effector position tracking")

        return axes

    def plot_base_path(self):
        """Plot base path."""
        r_b = self.data.get("r_bw_ws", [])

        if len(r_b) == 0:
            return

        _, ax = plt.subplots(1, 1)

        if len(r_b) > 0:
            r_b = np.array(r_b)  # Convert to numpy array
            ax.plot(r_b[:, 0], r_b[:, 1], label=self.data["name"], linewidth=1)

        ax.grid()
        ax.legend()
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Base Path Tracking")

    def plot_tracking_err(self):
        """Plot tracking error."""
        ts = self.data["ts"]

        _, ax = plt.subplots(1, 1)

        ax.plot(
            ts,
            self.data["err_base"],
            label=self.data["name"]
            + f" base err, rms={self.data['statistics']['err_base']['rms']:.3f}",
            linestyle="--",
        )
        ax.plot(
            ts,
            self.data["err_ee"],
            label=self.data["name"]
            + f" EE err, rms={self.data['statistics']['err_ee']['rms']:.3f}",
            linestyle="-",
        )

        ax.grid()
        ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (m)")
        ax.set_title("Tracking Error vs Time")

    def plot_task_performance(self):
        """Plot task performance metrics."""
        f, axes = plt.subplots(4, 1, sharex=True)

        legend = self.data["name"]
        t_sim = self.data["ts"]

        axes[0].plot(
            t_sim,
            self.data["constraints_violation"] * 100,
            label=f"{legend} mean={self.data['statistics']['constraints_violation']['mean']*100:.1f}%",
        )
        axes[0].set_ylabel("Constraints violation (%)")

        axes[1].plot(
            t_sim,
            self.data["err_ee"],
            label=f"{legend} acc={self.data['statistics']['err_ee']['integral']:.3f}",
        )
        axes[1].set_ylabel("EE Error (m)")

        axes[2].plot(
            t_sim,
            self.data["err_base"],
            label=f"{legend} acc={self.data['statistics']['err_base']['integral']:.3f}",
        )
        axes[2].set_ylabel("Base Error (m)")

        axes[3].plot(t_sim, self.data["arm_manipulability"], label=legend)
        axes[3].set_ylabel("Arm Manipulability")
        axes[3].set_xlabel("Time (s)")

        for ax in axes:
            ax.legend()
            ax.grid(True)
