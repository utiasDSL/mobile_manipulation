"""
Core plotting functionality for mobile manipulation experiments.

This module provides the main DataPlotter class and utility functions for
loading and processing simulation data.
"""

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from spatialmath.base import rotz

import mm_control.MPC as MPC
from mm_utils import math, parsing
from mm_utils.math import wrap_pi_array
from mm_utils.plotting.plot_mpc import MPCPlotterMixin
from mm_utils.plotting.plot_trajectory import TrajectoryPlotterMixin

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def construct_logger(
    path_to_folder,
    process=True,
    data_plotter_class=None,
):
    """Load data from an experiment folder.

    Expected folder structure:
        <path_to_folder>/
            sim/
                data.npz
            control/
                data.npz
                config.yaml

    For combined sim+control runs (single process):
        <path_to_folder>/
            combined/
                data.npz
                config.yaml

    Args:
        path_to_folder (str): Path to experiment folder.
        process (bool): If True, post-process data (compute tracking errors, etc.).
        data_plotter_class (class, optional): DataPlotter class to use. Defaults to DataPlotter.

    Returns:
        DataPlotter: Configured DataPlotter instance with loaded data.

    Raises:
        ValueError: If folder structure is unrecognized.
    """
    if data_plotter_class is None:
        data_plotter_class = DataPlotter

    items = set(os.listdir(path_to_folder))

    if "sim" in items and "control" in items:
        # ROS nodes: separate sim/ and control/ folders
        return data_plotter_class.from_ROSSIM_results(path_to_folder, process)
    elif len(items) == 1:
        # Combined run: single subfolder (e.g., combined/)
        subfolder = list(items)[0]
        return data_plotter_class.from_PYSIM_results(
            os.path.join(path_to_folder, subfolder), process
        )
    elif "data.npz" in items and "config.yaml" in items:
        # Direct data files in folder
        return data_plotter_class.from_PYSIM_results(path_to_folder, process)
    else:
        raise ValueError(f"Unrecognized folder structure in {path_to_folder}.")


# =============================================================================
# CORE DATAPLOTTER CLASS
# =============================================================================


class DataPlotter(TrajectoryPlotterMixin, MPCPlotterMixin):
    """Core class for loading and processing simulation data for plotting."""

    def __init__(self, data, config=None, process=True):
        """Initialize DataPlotter with simulation data.

        Args:
            data (dict): Dictionary containing simulation/control data.
            config (dict, optional): Configuration dictionary. If provided, enables controller-dependent processing.
            process (bool): If True, run post-processing and statistics computation.
        """
        self.data = data
        self.data["name"] = self.data.get("name", "data")
        self.name = self.data["name"]
        self.config = config
        if config is not None:
            # controller
            control_class = getattr(MPC, config["controller"]["type"], None)
            if control_class is None:
                raise ValueError(
                    f"Unknown controller type: {config['controller']['type']}"
                )

            config["controller"]["acados"]["cython"]["enabled"] = True
            config["controller"]["acados"]["cython"]["recompile"] = False
            self.controller = control_class(config["controller"])
            self.model_interface = self.controller.model_interface

        if process:
            self._post_processing()
            self._get_statistics()

    @classmethod
    def from_logger(cls, logger, process):
        """Create DataPlotter from DataLogger instance.

        Args:
            logger (DataLogger): DataLogger instance to convert.
            process (bool): If True, run post-processing and statistics computation.

        Returns:
            DataPlotter: DataPlotter instance with converted data.
        """
        # convert logger data to numpy format
        data = {}
        for key, value in logger.data.items():
            data[key] = np.array(value)
        return cls(data, process=process)

    @classmethod
    def from_npz(cls, npz_file_path, process):
        """Create DataPlotter from NPZ file.

        Args:
            npz_file_path (str): Path to NPZ file containing data.
            process (bool): If True, run post-processing and statistics computation.

        Returns:
            DataPlotter: DataPlotter instance with loaded data.
        """
        data = dict(np.load(npz_file_path))
        if "name" not in data:
            path_split = npz_file_path.split("/")
            folder_name = path_split[-2]
            data_name = folder_name.split("_")[:-2]
            data_name = "_".join(data_name)
            data["name"] = data_name
        return cls(data, process=process)

    @classmethod
    def from_PYSIM_results(cls, folder_path, process):
        """For data obtained from running controller in the simulation loop.

        Args:
            folder_path (str): Path to folder containing data.npz and config.yaml.
            process (bool): If True, post-process data (compute tracking errors, etc.).

        Returns:
            DataPlotter: Configured DataPlotter instance with loaded data.
        """
        npz_file_path = os.path.join(folder_path, "data.npz")
        data = dict(np.load(npz_file_path, allow_pickle=True))
        config_file_path = os.path.join(folder_path, "config.yaml")
        config = parsing.load_config(config_file_path)
        folder_name = folder_path.split("/")[-1]
        data["name"] = folder_name.split("_")[0]
        data["folder_path"] = folder_path

        return cls(data, config, process=process)

    @classmethod
    def from_ROSSIM_results(cls, folder_path, process):
        """For data from running simulation and controller as two ROS nodes.

        Args:
            folder_path (str): Path to folder containing sim/ and control/ subdirectories.
            process (bool): If True, post-process data (compute tracking errors, etc.).

        Returns:
            DataPlotter: Configured DataPlotter instance with loaded data.
        """
        data_decoupled = {}
        config = None

        sim_path = os.path.join(folder_path, "sim", "data.npz")
        control_path = os.path.join(folder_path, "control", "data.npz")
        config_path = os.path.join(folder_path, "control", "config.yaml")

        data_decoupled["sim"] = dict(np.load(sim_path, allow_pickle=True))
        data_decoupled["control"] = dict(np.load(control_path, allow_pickle=True))
        config = parsing.load_config(config_path)

        data = data_decoupled["control"]

        t = data["ts"]
        t_sim = data_decoupled["sim"]["ts"]
        for key, value in data_decoupled["sim"].items():
            if key in [
                "ts",
                "sim_timestep",
                "nq",
                "nv",
                "nx",
                "nu",
                "duration",
                "dir_path",
                "cmd_vels",
            ]:
                continue
            else:
                value = np.array(value)
                f_interp = interp1d(t_sim, value, axis=0, fill_value="extrapolate")
                data[key] = f_interp(t)
        data["ts"] -= data["ts"][0]
        data["name"] = folder_path.split("/")[-1]
        data["folder_path"] = folder_path

        return cls(data, config, process)

    def _get_tracking_err(self, ref_name, robot_traj_name):
        """Compute tracking error between reference and robot trajectories.

        Args:
            ref_name (str): Key name for reference trajectory in data dictionary.
            robot_traj_name (str): Key name for robot trajectory in data dictionary.

        Returns:
            ndarray: Tracking error norms over time.
        """
        N = len(self.data["ts"])
        rs = self.data.get(robot_traj_name, None)
        rds = self.data.get(ref_name, None)

        if rds is None:
            return np.zeros(N)
        if rs is None:
            rs = np.zeros_like(rds)

        if len(rs) == len(rds):
            # Handle dimension mismatch - for EE tracking, compare only position (first 3 elements)
            if rs.shape[1] != rds.shape[1]:
                # Take minimum dimensions to compare (typically position only)
                min_dim = min(rs.shape[1], rds.shape[1])
                rs_pos = rs[:, :min_dim]
                rds_pos = rds[:, :min_dim]
                errs = np.linalg.norm(rds_pos - rs_pos, axis=1)
            else:
                errs = np.linalg.norm(rds - rs, axis=1)
        else:
            errs = np.zeros(len(rs))
        return errs

    def _transform_w2b_SE3(self, qb, r_w):
        """Transform 3D points from world frame to base frame (SE2).

        Args:
            qb (ndarray): Base pose [x, y, yaw].
            r_w (ndarray): Points in world frame, shape (N, 3).

        Returns:
            ndarray: Points in base frame, shape (N, 3).
        """
        Rbw = rotz(-qb[2])
        rbw = np.array([qb[0], qb[1], 0])
        r_b = (Rbw @ (r_w - rbw).T).T

        return r_b

    def _transform_w2b_SE2(self, qb, r_w):
        """Transform 2D points from world frame to base frame (SE2).

        Args:
            qb (ndarray): Base pose [x, y, yaw].
            r_w (ndarray): Points in world frame, shape (N, 2).

        Returns:
            ndarray: Points in base frame, shape (N, 2).
        """
        Rbw = rotz(-qb[2])[:2, :2]
        rbw = np.array(qb[:2])
        r_b = (Rbw @ (r_w - rbw).T).T

        return r_b

    def _get_mean_violation(self, data_normalized):
        """Compute mean constraint violation from normalized data.

        Args:
            data_normalized (ndarray): Normalized data (values > 1 indicate violations).

        Returns:
            tuple: (vio_mean, total_violations) where vio_mean is mean violation per time step and total_violations is total count.
        """
        vio_mask = data_normalized > 1.05
        vio = np.sum((data_normalized - 1) * vio_mask, axis=1)
        vio_num = np.sum(vio_mask, axis=1)
        vio_mean = np.where(vio_num > 0, vio / vio_num, 0)
        return vio_mean, np.sum(vio_num)

    def _post_processing(self):
        """Post-process data to compute tracking errors, constraints, manipulability, and coordinate transforms."""
        # tracking error
        self.data["err_ee"] = self._get_tracking_err("r_ew_w_ds", "r_ew_ws")
        self.data["err_base"] = self._get_tracking_err("r_bw_w_ds", "r_bw_ws")
        self.data["err_ee_normalized"] = self.data["err_ee"] / self.data["err_ee"][0]
        self.data["err_base_normalized"] = (
            self.data["err_base"] / self.data["err_base"][0]
        )

        # signed distance
        nq = self.data["nq"]
        qs = self.data["xs"][:, :nq]

        # keyed by obstacle names or "self"
        names = ["self", "static_obstacles"]
        params = {"self": [], "static_obstacles": []}
        sds_dict = self.model_interface.evaluateSignedDistance(
            names, qs, copy.deepcopy(params)
        )
        sds = np.array([sd for sd in sds_dict.values()])
        for id, name in enumerate(names):
            self.data["_".join(["signed_distance", name])] = sds_dict[name]
        self.data["signed_distance"] = np.min(sds, axis=0)

        names = []
        params = {}

        if self.config["controller"]["self_collision_avoidance_enabled"]:
            names += ["self"]
            params = {"self": []}

        if self.config["controller"]["static_obstacles_collision_avoidance_enabled"]:
            params["static_obstacles"] = []
            names += ["static_obstacles"]
        sds_dict_detailed = self.model_interface.evaluateSignedDistancePerPair(
            names, qs, params
        )
        self.data["signed_distance_detailed"] = {}

        for name, sds in sds_dict_detailed.items():
            self.data["signed_distance_detailed"][name] = {}
            for pair, sd in sds.items():
                self.data["signed_distance_detailed"][name][pair] = sd

        # normalized state and input w.r.t bounds
        # -1 --> saturate lower bounds
        # 1  --> saturate upper bounds
        # 0  --> in middle
        bounds = self.config["controller"]["robot"]["limits"]
        self.data["xs_normalized"] = math.normalize_wrt_bounds(
            parsing.parse_array(bounds["state"]["lower"]),
            parsing.parse_array(bounds["state"]["upper"]),
            self.data["xs"],
        )
        self.data["cmd_vels_normalized"] = math.normalize_wrt_bounds(
            parsing.parse_array(bounds["state"]["lower"])[nq:],
            parsing.parse_array(bounds["state"]["upper"])[nq:],
            self.data["cmd_vels"],
        )
        self.data["cmd_accs_normalized"] = math.normalize_wrt_bounds(
            parsing.parse_array(bounds["input"]["lower"]),
            parsing.parse_array(bounds["input"]["upper"]),
            self.data["cmd_accs"],
        )

        # box constraints
        constraints_violation = np.abs(
            np.hstack((self.data["xs_normalized"], self.data["cmd_accs_normalized"]))
        )
        constraints_violation = np.hstack(
            (
                constraints_violation,
                np.expand_dims(0.05 - self.data["signed_distance"], axis=1) / 0.05 + 1,
            )
        )
        (
            self.data["constraints_violation"],
            self.data["constraints_violation_num"],
        ) = self._get_mean_violation(constraints_violation)

        # singularity
        man_fcn = self.model_interface.robot.manipulability_fcn
        man_fcn_map = man_fcn.map(len(self.data["ts"]))
        manipulability = man_fcn_map(self.data["xs"][:, :nq].T)
        self.data["manipulability"] = manipulability.toarray().flatten()

        arm_man_fcn = self.model_interface.robot.arm_manipulability_fcn
        arm_man_fcn_map = arm_man_fcn.map(len(self.data["ts"]))
        arm_manipulability = arm_man_fcn_map(self.data["xs"][:, :nq].T)
        self.data["arm_manipulability"] = arm_manipulability.toarray().flatten()

        # jerk
        self.data["cmd_jerks"] = (
            self.data["cmd_accs"][1:, :] - self.data["cmd_accs"][:-1, :]
        ) / np.expand_dims(self.data["ts"][1:] - self.data["ts"][:-1], axis=1)

        # coordinate transform
        qb = self.data["xs"][0, :3]

        # Transform only position components (first 3 dimensions)
        self.data["r_ew_bs"] = self._transform_w2b_SE3(qb, self.data["r_ew_ws"][:, :3])
        if "r_ew_w_ds" in self.data.keys():
            self.data["r_ew_b_ds"] = self._transform_w2b_SE3(
                qb, self.data["r_ew_w_ds"][:, :3]
            )

        # has_rb = self.data.get("r_bw_w_ds", None)
        self.data["r_bw_bs"] = self._transform_w2b_SE2(qb, self.data["r_bw_ws"])
        if "r_bw_w_ds" in self.data.keys():
            self.data["r_bw_b_ds"] = self._transform_w2b_SE2(qb, self.data["r_bw_w_ds"])
        if "yaw_bw_w_ds" in self.data.keys():
            self.data["yaw_bw_w_ds"] -= qb[2]
            self.data["yaw_bw_w_ds"] = wrap_pi_array(self.data["yaw_bw_w_ds"])
        if "yaw_bw_ws" in self.data.keys():
            self.data["yaw_bw_ws"] -= qb[2]
            self.data["yaw_bw_ws"] = wrap_pi_array(self.data["yaw_bw_ws"])

        N = len(self.data["ts"])

        self.data["mpc_ee_predictions"] = []
        self.data["mpc_base_predictions"] = []

        for t_index in range(N):
            x_bar = self.data["mpc_x_bars"][t_index]
            ee_bar, base_bar = self.controller._getEEBaseTrajectories(x_bar)
            self.data["mpc_ee_predictions"].append(ee_bar)
            self.data["mpc_base_predictions"].append(base_bar)

        self.data["mpc_ee_predictions"] = np.array(self.data["mpc_ee_predictions"])
        self.data["mpc_base_predictions"] = np.array(self.data["mpc_base_predictions"])

    def _get_statistics(self):
        """Compute statistics (RMS, integrals, mean, max, min, std) for tracking errors and other metrics."""
        self.data["statistics"] = {}
        # EE tracking error
        err_ee_stats = math.statistics(self.data["err_ee"])
        self.data["statistics"]["err_ee"] = {
            "rms": math.rms_continuous(self.data["ts"], self.data["err_ee"]),
            "integral": math.integrate_zoh(self.data["ts"], self.data["err_ee"]),
            "mean": err_ee_stats[0],
            "max": err_ee_stats[1],
            "min": err_ee_stats[2],
            "std": math.statistics_std(self.data["err_ee"]),
        }
        # base tracking error
        err_base_stats = math.statistics(self.data["err_base"])
        self.data["statistics"]["err_base"] = {
            "rms": math.rms_continuous(self.data["ts"], self.data["err_base"]),
            "integral": math.integrate_zoh(self.data["ts"], self.data["err_base"]),
            "mean": err_base_stats[0],
            "max": err_base_stats[1],
            "min": err_base_stats[2],
        }

        # EE tracking error (Normalized)
        err_ee_normalized_stats = math.statistics(self.data["err_ee_normalized"])
        self.data["statistics"]["err_ee_normalized"] = {
            "rms": math.rms_continuous(self.data["ts"], self.data["err_ee_normalized"]),
            "integral": math.integrate_zoh(
                self.data["ts"], self.data["err_ee_normalized"]
            ),
            "mean": err_ee_normalized_stats[0],
            "max": err_ee_normalized_stats[1],
            "min": err_ee_normalized_stats[2],
        }
        # base tracking error (Normalized)
        err_base_normalized_stats = math.statistics(self.data["err_base_normalized"])
        self.data["statistics"]["err_base_normalized"] = {
            "rms": math.rms_continuous(
                self.data["ts"], self.data["err_base_normalized"]
            ),
            "integral": math.integrate_zoh(
                self.data["ts"], self.data["err_base_normalized"]
            ),
            "mean": err_base_normalized_stats[0],
            "max": err_base_normalized_stats[1],
            "min": err_base_normalized_stats[2],
        }

        # signed distance
        sd_stats = math.statistics(self.data["signed_distance"])
        self.data["statistics"]["signed_distance"] = {
            "mean": sd_stats[0],
            "max": sd_stats[1],
            "min": sd_stats[2],
        }

        # bounds saturation
        nq = self.data["nq"]
        q_stats = math.statistics(np.abs(self.data["xs_normalized"][:, :nq].flatten()))
        self.data["statistics"]["q_saturation"] = {
            "mean": q_stats[0],
            "max": q_stats[1],
            "min": q_stats[2],
        }

        qdot_stats = math.statistics(
            np.abs(self.data["xs_normalized"][:, nq:].flatten())
        )
        self.data["statistics"]["qdot_saturation"] = {
            "mean": qdot_stats[0],
            "max": qdot_stats[1],
            "min": qdot_stats[2],
        }

        cmd_vels_stats = math.statistics(
            np.abs(self.data["cmd_vels_normalized"].flatten())
        )
        self.data["statistics"]["cmd_vels_saturation"] = {
            "mean": cmd_vels_stats[0],
            "max": cmd_vels_stats[1],
            "min": cmd_vels_stats[2],
        }

        cmd_accs_stats = math.statistics(
            np.abs(self.data["cmd_accs_normalized"].flatten())
        )
        self.data["statistics"]["cmd_accs_saturation"] = {
            "mean": cmd_accs_stats[0],
            "max": cmd_accs_stats[1],
            "min": cmd_accs_stats[2],
        }

        cmd_jerks_base_linear_stats = math.statistics(
            np.linalg.norm(self.data["cmd_jerks"][:, :2], axis=1).flatten()
        )
        self.data["statistics"]["cmd_jerks_base_linear"] = {
            "mean": cmd_jerks_base_linear_stats[0],
            "max": cmd_jerks_base_linear_stats[1],
            "min": cmd_jerks_base_linear_stats[2],
        }
        cmd_jerks_base_angular_stats = math.statistics(
            np.abs(self.data["cmd_jerks"][:, 2])
        )
        self.data["statistics"]["cmd_jerks_base_angular"] = {
            "mean": cmd_jerks_base_angular_stats[0],
            "max": cmd_jerks_base_angular_stats[1],
            "min": cmd_jerks_base_angular_stats[2],
        }

        cmd_jerks_stats = math.statistics(self.data["cmd_jerks"])
        self.data["statistics"]["cmd_jerks"] = {
            "mean": cmd_jerks_stats[0],
            "max": cmd_jerks_stats[1],
            "min": cmd_jerks_stats[2],
        }

        violation_stats = math.statistics(self.data["constraints_violation"])
        self.data["statistics"]["constraints_violation"] = {
            "mean": violation_stats[0],
            "max": violation_stats[1],
            "min": violation_stats[2],
            "num": self.data["constraints_violation_num"],
        }
        run_time_states = math.statistics(self.data["controller_run_time"])
        self.data["statistics"]["run_time"] = {
            "mean": run_time_states[0],
            "max": run_time_states[1],
            "min": run_time_states[2],
        }

    def summary(self, stat_names):
        """Get a summary of statistics.

        Args:
            stat_names (list): List of stats of interest, (key, value) pairs.

        Returns:
            list: Array of statistics.
        """

        stats = []
        stats_dict = self.data["statistics"]

        # Return None if either key or val doesn't exist
        for key, val in stat_names:
            stats.append(stats_dict.get(key, {}).get(val, None))

        return stats

    # Convenience methods for common plotting workflows
    def plot_all(self):
        """Plot all available data."""
        self.plot_tracking()
        self.plot_mpc()

    def plot_tracking(self):
        """Plot tracking performance."""
        self.plot_ee_tracking()
        self.plot_base_path()
        self.plot_tracking_err()
        self.plot_task_performance()
        plt.show()

    def plot_mpc(self):
        """Plot MPC-specific data."""
        self.plot_cost()
        self.plot_run_time()
        plt.show()
