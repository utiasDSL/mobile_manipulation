"""
MPC-specific plotting functionality.

This module provides plotting methods for Model Predictive Control (MPC) data,
including cost function visualization and controller execution time.
"""

import matplotlib.pyplot as plt
import numpy as np


class MPCPlotterMixin:
    """Mixin class containing MPC-specific plotting methods."""

    def plot_cost(self):
        """Plot MPC cost function over time."""
        t_sim = self.data["ts"]
        cost_final = self.data.get("mpc_cost_finals")

        if cost_final is None:
            print("No cost data found")
            return

        f, ax = plt.subplots(1, 1)

        # Convert to numpy array and handle different shapes
        cost_final = np.array(cost_final)
        if cost_final.ndim == 1:
            ax.plot(
                t_sim,
                cost_final,
                ".-",
                label=self.data["name"],
                linewidth=2,
                markersize=8,
            )
        else:
            # Multi-dimensional cost - plot first dimension
            ax.plot(
                t_sim,
                cost_final[:, 0],
                ".-",
                label=self.data["name"],
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cost")
        ax.legend()
        ax.set_title("MPC Cost")
        ax.grid(True)

    def plot_run_time(self):
        """Plot controller execution time."""
        t_sim = self.data["ts"]
        run_time = self.data.get("controller_run_time")
        if run_time is None:
            print("Ignore run time")
            return

        f, ax = plt.subplots(1, 1)
        ax.plot(t_sim, run_time * 1000, label=self.data["name"], linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("run time (ms)")
        ax.legend()
        ax.set_title("Controller Run Time")
