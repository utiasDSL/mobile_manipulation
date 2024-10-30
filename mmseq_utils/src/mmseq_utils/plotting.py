import numpy as np
from scipy.interpolate import interp1d
import copy
from typing import List

import matplotlib.pyplot as plt
# import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import yaml
import rosbag
import os
from spatialmath.base import rotz, r2q
import time
from tf.transformations import euler_from_quaternion

from mmseq_utils.parsing import parse_path, load_config
from mmseq_utils.casadi_struct import casadi_sym_struct, reconstruct_sym_struct_map_from_array
from mmseq_utils.math import wrap_pi_array

from mmseq_control.robot import CasadiModelInterface, MobileManipulator3D
from mmseq_control_new.MPCConstraints import SignedDistanceConstraint,SignedDistanceConstraintCBF
from mmseq_control_new.MPC import STMPC
import mmseq_control_new.MPC as MPC
import mmseq_control_new.HTMPC as HTMPC
import mmseq_control_new.HybridMPC as HybridMPC
from mmseq_utils import parsing, math
from matplotlib.backends.backend_pdf import PdfPages
from mobile_manipulation_central import ros_utils

VICON_TOOL_NAME = "ThingWoodLumber"
THING_BASE_NAME = "ThingBase_3"

SCREEN_WIDTH_PX = 2560  # Replace with your screen width in pixels
SCREEN_HEIGHT_PX = 1600  # Replace with your screen height in pixels
DPI = 96  # Typical screen DPI (dots per inch). Adjust if needed.

# Calculate the figure size in inches
FULL_SCREEN_WIDTH_INCH = SCREEN_WIDTH_PX / DPI
FULL_SCREEN_HEIGHT_INCH = SCREEN_HEIGHT_PX / DPI

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.set_figheight(FULL_SCREEN_HEIGHT_INCH)
        fig.set_figwidth(FULL_SCREEN_WIDTH_INCH)

        fig.savefig(pp, format='pdf', dpi=96)
    pp.close()

    
class DataPlotter:
    def __init__(self, data, config=None, process=True):
        self.data = data
        self.data["name"] = self.data.get('name', 'data')
        self.name = self.data["name"]
        self.config = config
        if config is not None:
            # controller
            control_class = getattr(MPC, config["controller"]["type"], None)
            if control_class is None:
                control_class = getattr(HTMPC, config["controller"]["type"], None)

            if control_class is None:
                control_class = getattr(HybridMPC, config["controller"]["type"], None)

            config["controller"]["acados"]["cython"]["enabled"] = True
            config["controller"]["acados"]["cython"]["action"] = "load"
            self.controller = control_class(config["controller"])
            self.model_interface = self.controller.model_interface

        if process:
            self._post_processing()
            self._get_statistics()

    @classmethod
    def from_logger(cls, logger, process):
        # convert logger data to numpy format
        data = {}
        for key, value in logger.data.items():
            data[key] = np.array(value)
        return cls(data, process=process)

    @classmethod
    def from_npz(cls, npz_file_path, process):
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
        """ For data obtained from running controller in the simulation loop

        :param folder_path:
        :return:
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
        """ for data obtained from running simulation and controller as two nodes

        :param folder_path:
        :return:
        """
        data_decoupled = {}
        config = None
        for filename in os.listdir(folder_path):
            d = os.path.join(folder_path, filename)
            key = filename.split("_")[0]
            if os.path.isdir(d):
                path_to_npz = os.path.join(d, "data.npz")
                data_decoupled[key] = dict(np.load(path_to_npz, allow_pickle=True))

            if key == 'control':
                path_to_config = os.path.join(d, "config.yaml")
                config = parsing.load_config(path_to_config)

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
        data["folder_path"] = folder_path

        return cls(data, config, process)

    @classmethod
    def from_ROSEXP_results(cls, folder_path, process):
        """ for data obtained in experiments, robot data comes from ROS bag and controller data collected by DataLogger

        :param folder_path: path to folder that contains the control_ data folder and rosbag
        :return:
        """
        config = None

        for filename in os.listdir(folder_path):
            d = os.path.join(folder_path, filename)

            if os.path.isdir(d):
                path_to_control_folder = d
            else:
                path_to_bag = d

        path_to_npz = os.path.join(path_to_control_folder, "data.npz")
        data = dict(np.load(path_to_npz, allow_pickle=True))

        path_to_config = os.path.join(path_to_control_folder, "config.yaml")
        config = parsing.load_config(path_to_config)

        ros_bag_plotter = ROSBagPlotter(path_to_bag, path_to_config)
        ros_data = ros_bag_plotter.data
        xs = np.hstack((ros_data["ridgeback"]["joint_states"]["qs"],
                        ros_data["ur10"]["joint_states_interpolated"]["qs"],
                        ros_data["ridgeback"]["joint_states"]["vs"],
                        ros_data["ur10"]["joint_states_interpolated"]["vs"],
                        ))
        r_ew_ws = ros_data["model"]["EE"]["pos"]
        q_ew_ws = ros_data["model"]['EE']["orn"]
        v_ew_ws = ros_data["model"]["EE"]["vel_lin"]
        ω_ew_ws = ros_data["model"]["EE"]["vel_ang"]


        r_bw_ws = ros_data["model"]['base']["pos"]
        yaw_bw_ws = ros_data["model"]['base']["orn"]
        v_bw_ws = ros_data["model"]["base"]["vel_lin"]
        ω_bw_ws = ros_data["model"]["base"]["vel_ang"]

        values = [xs, r_ew_ws, q_ew_ws,v_ew_ws,ω_ew_ws, r_bw_ws, yaw_bw_ws,v_bw_ws,ω_bw_ws]
        ts = [ros_data["ridgeback"]["joint_states"]["ts"]] + [ros_data["model"]["EE"]["ts"]] *4 + [ros_data["model"]["base"]["ts"]] * 4
        keys = ["xs", "r_ew_ws", "q_ew_ws", "v_ew_ws", "ω_ew_ws", "r_bw_ws", "yaw_bw_ws", "v_bw_ws", "ω_bw_ws"]

        # raw robot data from rosbag, keep both interpolated and raw data
        data["raw"] = {}
        for t, value, key in zip(ts, values, keys):
            value = np.array(value)
            f_interp = interp1d(t, value, axis=0, fill_value="extrapolate")
            data[key] = f_interp(data["ts"])
            data["raw"][key] = {"ts": t, "value": value}

        for key in keys:
            data["raw"][key]["ts"] -= data["ts"][0]
        data["ts"] -= data["ts"][0]
        
        data["name"] = folder_path.split("/")[-1]
        data["folder_path"] = folder_path
        return cls(data, config, process)

    def _get_tracking_err(self, ref_name, robot_traj_name):
        N = len(self.data["ts"])
        rs = self.data.get(robot_traj_name, None)
        rds = self.data.get(ref_name, None) 
        
        if rds is None:
            return np.zeros(N)
        if rs is None:
            rs = np.zeros_like(rds)

        if len(rs) == len(rds):
            errs = np.linalg.norm(rds - rs, axis=1)
        else:
            errs = np.zeros(len(rs))
        return errs

    def _transform_w2b_SE3(self, qb, r_w):
        Rbw = rotz(-qb[2])
        rbw = np.array([qb[0], qb[1], 0])
        r_b = (Rbw @ (r_w - rbw).T).T    

        return r_b

    def _transform_w2b_SE2(self, qb, r_w):
        Rbw = rotz(-qb[2])[:2, :2]
        rbw = np.array(qb[:2])
        r_b = (Rbw @ (r_w - rbw).T).T    

        return r_b

    def _get_mean_violation(self, data_normalized):
        vio_mask = data_normalized > 1.05
        vio = np.sum((data_normalized - 1) * vio_mask, axis=1)
        vio_num = np.sum(vio_mask, axis=1)
        vio_mean = np.where(vio_num > 0, vio/vio_num, 0)
        return vio_mean, np.sum(vio_num)

    def _post_processing(self):
        # tracking error
        self.data["err_ee"] = self._get_tracking_err("r_ew_w_ds", "r_ew_ws")
        self.data["err_base"] = self._get_tracking_err("r_bw_w_ds", "r_bw_ws")
        self.data["err_ee_normalized"] = self.data["err_ee"]/self.data["err_ee"][0]
        self.data["err_base_normalized"] = self.data["err_base"]/self.data["err_base"][0]

        # signed distance
        nq = self.data["nq"]
        qs = self.data["xs"][:, :nq]

        print(self.data["xs"].shape)

        # keyed by obstacle names or "self"
        names = ["self", "static_obstacles"]
        params = {"self": [], "static_obstacles":[]}
        if self.config["controller"]["sdf_collision_avoidance_enabled"]:
            names += ["sdf"]
            sdf_param_names = ["_".join(["mpc","sdf","param", str(i)])+"s" for i in range(self.model_interface.sdf_map.dim+1)]
            sdf_param = [self.data[name] for name in sdf_param_names]
            params["sdf"] = sdf_param
        sds_dict = self.model_interface.evaluteSignedDistance(names, qs, params)
        sds = np.array([sd for sd in sds_dict.values()])
        for id, name in enumerate(names):
            self.data["_".join(["signed_distance", name])] = sds_dict[name]
        self.data["signed_distance"] = np.min(sds, axis=0)

        # normalized state and input w.r.t bounds
        # -1 --> saturate lower bounds
        # 1  --> saturate upper bounds
        # 0  --> in middle
        bounds = self.config["controller"]["robot"]["limits"]
        self.data["xs_normalized"] = math.normalize_wrt_bounds(parsing.parse_array(bounds["state"]["lower"]),
                                                               parsing.parse_array(bounds["state"]["upper"]),
                                                               self.data["xs"])
        self.data["cmd_vels_normalized"] = math.normalize_wrt_bounds(parsing.parse_array(bounds["state"]["lower"])[nq:],
                                                                     parsing.parse_array(bounds["state"]["upper"])[nq:],
                                                                     self.data["cmd_vels"])
        self.data["cmd_accs_normalized"] = math.normalize_wrt_bounds(parsing.parse_array(bounds["input"]["lower"]),
                                                                     parsing.parse_array(bounds["input"]["upper"]),
                                                                     self.data["cmd_accs"])

        # self.data["xs_violation"] = self._get_mean_violation(np.abs(self.data["xs_normalized"]))
        # self.data["cmd_vels_violation"] = self._get_mean_violation(np.abs(self.data["cmd_vels_normalized"]))
        # self.data["cmd_accs_violation"] = self._get_mean_violation(np.abs(self.data["cmd_accs_normalized"]))
        # self.data["collision_violation"] = np.abs((self.data["signed_distance"] - 0.05)) / 0.05 * (self.data["signed_distance"] < 0.05)

        # box constraints
        constraints_violation = np.abs(np.hstack((self.data["xs_normalized"], self.data["cmd_accs_normalized"])))
        constraints_violation = np.hstack((constraints_violation, np.expand_dims(0.05 - self.data["signed_distance"], axis=1) / 0.05 + 1))
        self.data["constraints_violation"], self.data["constraints_violation_num"] = self._get_mean_violation(constraints_violation)

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
        self.data["cmd_jerks"] = (self.data["cmd_accs"][1:, :] - self.data["cmd_accs"][:-1, :]) / \
                                 np.expand_dims(self.data["ts"][1:] - self.data["ts"][:-1], axis=1)

        # coordinate transform
        qb = self.data["xs"][0, :3]

        self.data["r_ew_bs"] = self._transform_w2b_SE3(qb, self.data["r_ew_ws"])
        if "r_ew_w_ds" in self.data.keys():
            self.data["r_ew_b_ds"] = self._transform_w2b_SE3(qb, self.data["r_ew_w_ds"])

        # has_rb = self.data.get("r_bw_w_ds", None)
        self.data["r_bw_bs"] = self._transform_w2b_SE2(qb, self.data["r_bw_ws"])
        if "r_bw_w_ds" in self.data.keys():
            self.data["r_bw_b_ds"] = self._transform_w2b_SE2(qb, self.data["r_bw_w_ds"])
        if "yaw_bw_w_ds" in self.data.keys():
            self.data["yaw_bw_w_ds"] -= qb[2]
            self.data["yaw_bw_w_ds"] = wrap_pi_array(self.data["yaw_bw_w_ds"])
        if  "yaw_bw_ws" in self.data.keys():
            self.data["yaw_bw_ws"] -= qb[2]
            self.data["yaw_bw_ws"] = wrap_pi_array(self.data["yaw_bw_ws"])
        
        N = len(self.data["ts"])

        self.data["mpc_ee_predictions"] = []
        self.data["mpc_base_predictions"] = []

        for t_index in range(N):
            print(self.data["mpc_x_bars"].shape)
            x_bar = self.data["mpc_x_bars"][t_index]
            ee_bar, base_bar = self.controller._getEEBaseTrajectories(x_bar)
            self.data["mpc_ee_predictions"].append(ee_bar)
            self.data["mpc_base_predictions"].append(base_bar)
        
        self.data["mpc_ee_predictions"] = np.array(self.data["mpc_ee_predictions"])
        self.data["mpc_base_predictions"] = np.array(self.data["mpc_base_predictions"])


    def _get_statistics(self):
        self.data["statistics"] = {}
        # EE tracking error
        err_ee_stats = math.statistics(self.data["err_ee"])
        self.data["statistics"]["err_ee"] = {"rms": math.rms_continuous(self.data["ts"], self.data["err_ee"]),
                                             "integral": math.integrate_zoh(self.data["ts"], self.data["err_ee"]),
                                             "mean": err_ee_stats[0], "max": err_ee_stats[1], "min": err_ee_stats[2],
                                             "std": math.statistics_std(self.data["err_ee"])}
        # base tracking error
        err_base_stats = math.statistics(self.data["err_base"])
        self.data["statistics"]["err_base"] = {"rms": math.rms_continuous(self.data["ts"], self.data["err_base"]),
                                               "integral": math.integrate_zoh(self.data["ts"], self.data["err_base"]),
                                               "mean": err_base_stats[0], "max": err_base_stats[1], "min": err_base_stats[2]}

        # EE tracking error (Normalized)
        err_ee_normalized_stats = math.statistics(self.data["err_ee_normalized"])
        self.data["statistics"]["err_ee_normalized"] = {"rms": math.rms_continuous(self.data["ts"], self.data["err_ee_normalized"]),
                                             "integral": math.integrate_zoh(self.data["ts"], self.data["err_ee_normalized"]),
                                             "mean": err_ee_normalized_stats[0], "max": err_ee_normalized_stats[1], "min": err_ee_normalized_stats[2]}
        # base tracking error (Normalized)
        err_base_normalized_stats = math.statistics(self.data["err_base_normalized"])
        self.data["statistics"]["err_base_normalized"] = {"rms": math.rms_continuous(self.data["ts"], self.data["err_base_normalized"]),
                                               "integral": math.integrate_zoh(self.data["ts"], self.data["err_base_normalized"]),
                                               "mean": err_base_normalized_stats[0], "max": err_base_normalized_stats[1], "min": err_base_normalized_stats[2]}

        # signed distance
        sd_stats = math.statistics(self.data["signed_distance"])
        self.data["statistics"]["signed_distance"] = {"mean": sd_stats[0], "max": sd_stats[1], "min": sd_stats[2]}

        # bounds saturation
        nq = self.data["nq"]
        q_stats = math.statistics(np.abs(self.data["xs_normalized"][:, :nq].flatten()))
        self.data["statistics"]["q_saturation"] = {"mean": q_stats[0], "max": q_stats[1], "min": q_stats[2]}

        qdot_stats = math.statistics(np.abs(self.data["xs_normalized"][:, nq:].flatten()))
        self.data["statistics"]["qdot_saturation"] = {"mean": qdot_stats[0], "max": qdot_stats[1], "min": qdot_stats[2]}

        cmd_vels_stats = math.statistics(np.abs(self.data["cmd_vels_normalized"].flatten()))
        self.data["statistics"]["cmd_vels_saturation"] = {"mean": cmd_vels_stats[0], "max": cmd_vels_stats[1], "min": cmd_vels_stats[2]}

        cmd_accs_stats = math.statistics(np.abs(self.data["cmd_accs_normalized"].flatten()))
        self.data["statistics"]["cmd_accs_saturation"] = {"mean": cmd_accs_stats[0], "max": cmd_accs_stats[1], "min": cmd_accs_stats[2]}

        cmd_jerks_base_linear_stats = math.statistics(np.linalg.norm(self.data["cmd_jerks"][:, :2], axis=1).flatten())
        self.data["statistics"]["cmd_jerks_base_linear"] = {"mean": cmd_jerks_base_linear_stats[0], "max": cmd_jerks_base_linear_stats[1],
                                                          "min": cmd_jerks_base_linear_stats[2]}
        cmd_jerks_base_angular_stats = math.statistics(np.abs(self.data["cmd_jerks"][:, 2]))
        self.data["statistics"]["cmd_jerks_base_angular"] = {"mean": cmd_jerks_base_angular_stats[0],
                                                            "max": cmd_jerks_base_angular_stats[1],
                                                            "min": cmd_jerks_base_angular_stats[2]}

        cmd_jerks_stats = math.statistics(self.data["cmd_jerks"])
        self.data["statistics"]["cmd_jerks"] = {"mean": cmd_jerks_stats[0],
                                                            "max": cmd_jerks_stats[1],
                                                            "min": cmd_jerks_stats[2]}

        violation_stats = math.statistics(self.data["constraints_violation"])
        self.data["statistics"]["constraints_violation"] = {"mean": violation_stats[0], "max": violation_stats[1], "min": violation_stats[2], "num": self.data["constraints_violation_num"]}
        run_time_states = math.statistics(self.data["controller_run_time"])
        self.data["statistics"]["run_time"] = {"mean": run_time_states[0], "max": run_time_states[1], "min": run_time_states[2]}
        print(self.data["statistics"]["constraints_violation"]["num"])


    def summary(self, stat_names):
        """ get a summary of statistics

        :param stat_names: list of stats of interests, (key, value) pairs
        :return: array
        """

        stats = []
        stats_dict = self.data["statistics"]

        # Return None if either key or val doesn't exist
        for (key, val) in stat_names:
            stats.append(stats_dict.get(key, {}).get(val, None))

        return stats
    
    def _convert_np_array_to_dict(self, array):
        return dict(enumerate(array.flatten(), 1))[1]
    
    def run_mpc_iter(self, t):
        t_sim = self.data["ts"]
        t_index = np.argmin(np.abs(t_sim - t))

        iter_snapshot = self._convert_np_array_to_dict(self.data["mpc_iter_snapshots"][t_index])
        controller = self.controller
        ocp_solver = self.controller.ocp_solver

        u_bar_init = iter_snapshot["u_bar_init"]
        x_bar_init = iter_snapshot["x_bar_init"]
        p_map_bar = iter_snapshot["p_map_bar"]

        t0 = time.perf_counter()
        xo = iter_snapshot["xo"]+ np.array([0,0.0]+[0]*16)
        x_bar_init = controller._predictTrajectories(xo, u_bar_init)

        for i in range(self.controller.N+1):
            
            ocp_solver.set(i, 'x', x_bar_init[i])
            if i < controller.N:
                ocp_solver.set(i, 'u', u_bar_init[i])
            ocp_solver.set(i, 'p', p_map_bar[i])

        ocp_solver.solve_for_x0(xo, fail_on_nonzero_status=False)
        ocp_solver.print_statistics()

        x_bar = np.zeros_like(iter_snapshot["x_bar"])
        u_bar = np.zeros_like(iter_snapshot["u_bar"])

        for i in range(controller.N):
            x_bar[i,:] = ocp_solver.get(i, "x")
            u_bar[i,:] = ocp_solver.get(i, "u")
        x_bar[-1,:] = ocp_solver.get(controller.N, "x")

        x_bar_diff = x_bar - iter_snapshot["x_bar"]
        u_bar_diff = u_bar - iter_snapshot["u_bar"]
        t1 = time.perf_counter()

        print("x bar diff {}".format(x_bar_diff))
        print("u bar diff {}".format(u_bar_diff))
        print("u0 {}".format(u_bar[0]))
        print("run time {}".format(t1-t0))
        print("u0 {}".format(u_bar[0]))
        

    def plot_ee_tracking(self, axes=None, index=0, legend=None, base_frame=False):
        ts = self.data["ts"]
        r_ew_w_ds = self.data.get("r_ew_w_ds", [])
        N = len(ts)
        if base_frame:
            qb = self.data["xs"][0, :3]

            r_ew_ws = [self._transform_w2b_SE3(self.data["xs"][k, :3], 
                                                            self.data["r_ew_ws"][k]) for k in range(N)]
            r_ew_ws = np.array(r_ew_ws).reshape((N, 3))
            print(r_ew_ws)
        else:
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
        axes.set_title("End effector position tracking")

        return axes

    def plot_base_path(self, axes=None, index=0, legend=None, worldframe=True, linewidth=1, color=None):
        if worldframe:
            r_b = self.data.get("r_bw_ws", [])
        else:
            r_b = self.data.get("r_bw_bs", [])

        if len(r_b) == 0:
            return

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
            axes.plot(r_b[:, 0], r_b[:, 1], label=legend, color=color, linewidth=linewidth)

        axes.grid()
        axes.legend()
        axes.set_xlabel("x (m)")
        axes.set_ylabel("y (m)")
        axes.set_title("Base Path Tracking")

        return axes

    def plot_base_ref_path(self, axes=None, index=0, legend=None, worldframe=True, color='b'):
        if worldframe:
            r_b = self.data.get("r_bw_w_ds", [])
        else:
            r_b = self.data.get("r_bw_b_ds", [])

        if len(r_b) == 0:
            return

        if axes is None:
            axes = []
            f, axes = plt.subplots(1, 1, sharex=True)

        if legend is None:
            legend = self.data["name"]
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        if len(r_b) > 0:
            axes.plot(r_b[:, 0], r_b[:, 1], label=legend, color=color, linestyle="--")

        axes.grid()
        axes.legend()
        axes.set_xlabel("x (m)")
        axes.set_ylabel("y (m)")

        return axes

    def plot_ee_path(self, axes=None, index=0, legend=None):
        r_ew_ws = self.data.get("r_ew_ws", [])
        if len(r_ew_ws) == 0:
            return

        if axes is None:
            axes = []
            f, axes = plt.subplots(1, 1, sharex=True)

        if legend is None:
            legend = self.data["name"]
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        if len(r_ew_ws) > 0:
            axes.plot(r_ew_ws[:, 0], r_ew_ws[:, 1], label=legend + " EE path", color=colors[index])

        axes.grid()
        axes.legend()
        axes.set_xlabel("x (m)")
        axes.set_ylabel("y (m)")

        return axes

    def plot_ee_ref_path(self, axes=None, index=0, legend=None):
        r_ew_w_ds = self.data.get("r_ew_w_ds", [])
        if len(r_ew_w_ds) == 0:
            return

        if axes is None:
            axes = []
            f, axes = plt.subplots(1, 1, sharex=True)

        if legend is None:
            legend = self.data["name"]
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        if len(r_ew_w_ds) > 0:
            axes.plot(r_ew_w_ds[:, 0], r_ew_w_ds[:, 1], label=legend + " EE ref", color=colors[index],
                      linestyle="--")

        axes.grid()
        axes.legend()
        axes.set_xlabel("x (m)")
        axes.set_ylabel("y (m)")

        return axes

    def plot_ee_waypoints(self, axes=None, index=0, legend=None):
        r_ew_w_ds = self.data.get("r_ew_w_ds", [])
        if len(r_ew_w_ds) == 0:
            return

        if axes is None:
            axes = []
            f, axes = plt.subplots(1, 1, sharex=True)

        if legend is None:
            legend = self.data["name"]
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        if len(r_ew_w_ds) > 0:
            axes.scatter(r_ew_w_ds[:, 0], r_ew_w_ds[:, 1], label=legend + " EE waypoints", color=colors[index])

        axes.grid()
        axes.legend()
        axes.set_xlabel("x (m)")
        axes.set_ylabel("y (m)")

        return axes

    def plot_base_tracking(self, axes=None, index=0, legend=None):
        ts = self.data["ts"]
        r_ew_w_ds = self.data.get("r_bw_w_ds", [])
        r_ew_ws = self.data.get("r_bw_ws",[])
        yaw_bw_w_ds = self.data.get("yaw_bw_w_ds", [])
        yaw_bw_ws = self.data.get("yaw_bw_ws", [])

        if len(r_ew_w_ds) == 0 and len(r_ew_ws) == 0 and len(yaw_bw_w_ds) ==0 and len(yaw_bw_ws)==0:
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
        if len(yaw_bw_w_ds)>0:
            axes.plot(ts, yaw_bw_w_ds, label=legend + "$Θ_d$", color="b", linestyle="--" )
        if len(yaw_bw_ws)>0:
            axes.plot(ts, yaw_bw_ws, label=legend + "$Θ$", color="b" )

        axes.grid()
        axes.legend()
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Position (m)/Heading (rad)")
        axes.set_title("Base state tracking")

        return axes

    def plot_base_state_separate(self, axes=None, index=0, legend=None, linewidth=1, color=None):
        ts = self.data["ts"]
        r_ew_ws = self.data.get("r_bw_ws",[])
        yaw_bw_ws = self.data.get("yaw_bw_ws", [])

        if len(r_ew_ws) == 0 and len(yaw_bw_ws)==0:
            return

        if axes is None:
            axes = []
            f, axes = plt.subplots(3, 1, sharex=True)

        if legend is None:
            legend = self.data["name"]

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        # colors = sns.color_palette("ch:s=.35,rot=-.35", n_colors=3)
        cm = LinearSegmentedColormap.from_list(
            "my_list", colors, N=50)
        
        if color is None:
            color = cm(index)
            
        if len(r_ew_ws) > 0:
            for i in range(2):
                axes[i].plot(ts, r_ew_ws[:, i], label=" ".join([legend, "actual"]), color=color, linestyle="-",linewidth=linewidth)

        if len(yaw_bw_ws)>0:
            axes[2].plot(ts, yaw_bw_ws, label=" ".join([legend, "actual"]), color=color, linestyle="-",linewidth=linewidth )


        y_label = ["x", "y", "$Θ$"]
        for i, ax in enumerate(axes):
            ax.grid('on')
            ax.legend()
            ax.set_ylabel(y_label[i])

        axes[0].set_title("Base state tracking")
        axes[2].set_xlabel("Time (s)")

        return axes

    def plot_base_ref_separate(self, axes=None, index=0, legend=None, linewidth=1, color=None):
        ts = self.data["ts"]
        r_ew_w_ds = self.data.get("r_bw_w_ds", [])
        yaw_bw_w_ds = self.data.get("yaw_bw_w_ds", [])

        if len(r_ew_w_ds) == 0  and len(yaw_bw_w_ds) ==0:
            return

        if axes is None:
            axes = []
            f, axes = plt.subplots(3, 1, sharex=True)

        if legend is None:
            legend = self.data["name"]

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        # colors = sns.color_palette("ch:s=.35,rot=-.35", n_colors=3)
        cm = LinearSegmentedColormap.from_list(
            "my_list", colors, N=50)
        
        if color is None:
            color = cm(index)
            
        if len(r_ew_w_ds) > 0:
            for i in range(2):
                axes[i].plot(ts, r_ew_w_ds[:, i], label=legend + "desired", color="r", linestyle="--")
        if len(yaw_bw_w_ds)>0:
            axes[2].plot(ts, yaw_bw_w_ds, label=legend + "desired", color="r", linestyle="--",linewidth=linewidth )

        y_label = ["x", "y", "$Θ$"]
        for i, ax in enumerate(axes):
            ax.grid('on')
            ax.legend()
            ax.set_ylabel(y_label[i])

        axes[0].set_title("Base state tracking")
        axes[2].set_xlabel("Time (s)")

        return axes


    def plot_base_velocity_tracking(self, axes=None, index=0, legend=None):
        ts = self.data["ts"]
        v_bw_w_ds = self.data.get("v_bw_w_ds", [])
        v_bw_ws = self.data.get("v_bw_ws",[])
        ω_bw_w_ds = self.data.get("ω_bw_w_ds", [])
        ω_bw_ws = self.data.get("ω_bw_ws", [])

        if len(v_bw_w_ds) == 0 and len(v_bw_ws) == 0 and len(ω_bw_w_ds) ==0 and len(ω_bw_ws)==0:
            return

        if axes is None:
            axes = []
            f, axes = plt.subplots(1, 1, sharex=True)

        if legend is None:
            legend = self.data["name"]
        if len(v_bw_w_ds) > 0:
            axes.plot(ts, v_bw_w_ds[:, 0], label=legend + "${v_x}_d$", color="r", linestyle="--")
            axes.plot(ts, v_bw_w_ds[:, 1], label=legend + "${v_y}_d$", color="g", linestyle="--")
        if len(v_bw_ws) > 0:
            axes.plot(ts, v_bw_ws[:, 0], label=legend + "$v_x$", color="r")
            axes.plot(ts, v_bw_ws[:, 1], label=legend + "$v_y$", color="g")
        if len(ω_bw_w_ds)>0:
            axes.plot(ts, ω_bw_w_ds, label=legend + "$ω_d$", color="b", linestyle="--" )
        if len(ω_bw_ws)>0:
            axes.plot(ts, ω_bw_ws, label=legend + "$ω$", color="b" )

        axes.grid()
        axes.legend()
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Velocity (m/s or rad/s)")
        axes.set_title("Base velocity tracking")

        return axes


    def plot_ee_orientation_tracking(self, axes=None, index=0, legend=None):
        ts = self.data["ts"]
        Q_we_ds = self.data.get("Q_we_ds", None)
        Q_wes = self.data.get("Q_wes", None)

        if Q_wes is None or Q_we_ds is None:
            return
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

    def plot_ee_linear_velocity_tracking(self, axes=None, index=0, legend=None):
        ts = self.data["ts"]
        v_ew_ws = self.data["v_ew_ws"]
        # ω_ew_ws = self.data["ω_ew_ws"]

        v_ref = self.data.get("v_ew_w_ds", [])

        if axes is None:
            axes = []
            f, axes = plt.subplots(1, 1, sharex=True)
        if legend is None:
            legend = self.data["name"]

        axes.plot(ts, v_ew_ws[:, 0], label=legend + "$v_x$", color="r",)
        axes.plot(ts, v_ew_ws[:, 1], label=legend + "$v_y$", color="g",)
        axes.plot(ts, v_ew_ws[:, 2], label=legend + "$v_z$", color="b",)
        if len(v_ref) > 0:
            axes.plot(ts, v_ref[:, 0], label=legend + "${v_x}_d}$", color="r", linestyle="--")
            axes.plot(ts, v_ref[:, 1], label=legend + "${v_y}_d}$", color="g", linestyle="--")
            axes.plot(ts, v_ref[:, 2], label=legend + "${v_z}_d}$", color="b", linestyle="--")
        # axes.plot(ts, ω_ew_ws[:, 0], label=legend + "$ω_x$")
        # axes.plot(ts, ω_ew_ws[:, 1], label=legend + "$ω_y$")
        # axes.plot(ts, ω_ew_ws[:, 2], label=legend + "$ω_z$")
        axes.grid()
        axes.legend()
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Velocity")
        axes.set_title("End effector linear velocity tracking")

        return axes

    def plot_tracking_err(self, axes=None, index=0, legend=None):
        ts = self.data["ts"]
        if axes is None:
            axes = []
            f, axes = plt.subplots(1, 1, sharex=True)

        if legend is None:
            legend = self.data["name"]

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        axes.plot(ts, self.data["err_base"], label=legend+" $err_{base}, rms_{base} = $" + str(self.data["statistics"]["err_base"]["rms"]), linestyle="--", color=colors[index])
        axes.plot(ts, self.data["err_ee"], label=legend+" $err_{ee}, rms_{ee} = $" + str(self.data["statistics"]["err_ee"]["rms"]), linestyle="-", color=colors[index])

        axes.grid()
        axes.legend()
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Err (m)")
        axes.set_title("Tracking Error vs Time")

        return axes

    def plot_collision(self):
        nq = int(self.data["nq"])
        ts = self.data["ts"]
        qs = self.data["xs"][:, :nq]
        names = []
        params = {}

        if self.config["controller"]["self_collision_avoidance_enabled"]:
            names += ["self"]
            params = {"self": []}

        if self.config["controller"]["sdf_collision_avoidance_enabled"]:
            param_names =  ["_".join(["mpc","sdf", "param", str(i)])+"s" for i in range(self.model_interface.sdf_map.dim+1)]
            sdf_params = [self.data[name] for name in param_names]
            params["sdf"] = sdf_params
            names += ["sdf"]

        if self.config["controller"]["static_obstacles_collision_avoidance_enabled"]:
            params["static_obstacles"] = []
            names += ["static_obstacles"]

        sds = self.model_interface.evaluteSignedDistancePerPair(names, qs, params)

        axes = []
        for name, sd_per_pair in sds.items():
            f, ax = plt.subplots(1, 1)
            for pair, sd in sd_per_pair.items():
                ax.plot(ts, sd, label=pair)
            
            margin = self.config["controller"]["collision_safety_margin"].get(name, None)
            if margin is None:
                margin = self.config["controller"]["collision_safety_margin"].get("static_obstacles", None)

            if margin:
                ax.plot(ts, [margin]*len(ts), 'r--', linewidth=2, label=f"minimum clearance")

            ax.set_title("{} Distance".format(name))
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Sd(q) (m)")
            ax.legend()

    def plot_sdf_collision_separate(self, axes=None, index=0, color=None, linewidth=1, block=True, legend=None):
        nq = int(self.data["nq"])
        ts = self.data["ts"]
        qs = self.data["xs"][:, :nq]
        names = []
        params = {}

        if self.config["controller"]["sdf_collision_avoidance_enabled"]:
            param_names =  ["_".join(["mpc","sdf", "param", str(i)])+"s" for i in range(self.model_interface.sdf_map.dim+1)]
            sdf_params = [self.data[name] for name in param_names]
            params["sdf"] = sdf_params
            names += ["sdf"]

        sds = self.model_interface.evaluteSignedDistancePerPair(names, qs, params)
        num_pairs = 0
        for name, sd_per_pair in sds.items():
            for pair, sd in sd_per_pair.items():
                num_pairs += 1

        if axes is None:
            axes = []
            f, axes = plt.subplots(num_pairs, 1, sharex=True)

        if legend is None:
            legend = self.data["name"]

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        # colors = sns.color_palette("ch:s=.35,rot=-.35", n_colors=3)
        cm = LinearSegmentedColormap.from_list(
            "my_list", colors, N=50)

        
        if color is None:
            color = cm(index)

        i = 0
        for name, sd_per_pair in sds.items():
            for pair, sd in sd_per_pair.items():
                ax = axes[i]
                ax.plot(ts, sd, label=" ".join([legend, pair]), color=color, linestyle="-",linewidth=linewidth)
                ax.set_ylabel("Sd(q) (m)")
                ax.legend()
                ax.grid('on')

                i += 1
            
                margin = self.config["controller"]["collision_safety_margin"].get(name, None)
                if margin is None:
                    margin = self.config["controller"]["collision_safety_margin"].get("static_obstacles", None)

                if margin:
                    ax.plot(ts, [margin]*len(ts), 'r--', linewidth=2, label=f"minimum clearance")

            axes[0].set_title("{} Distance".format(name))
            axes[-1].set_xlabel("Time (s)")
        
        return axes

    def plot_collision_detailed(self):
        nq = int(self.data["nq"])
        ts = self.data["ts"]
        qs = self.data["xs"][:, 3:nq]
        names = []
        params = {}
        sds = []
        for q in qs:
            sd, ns = self.model_interface.pinocchio_interface.computeDistances(q)
            sds.append(sd)
        
        sds = np.array(sds)

        data_per_fig = 8  # Number of datasets per figure
        n_total = len(ns)  # Total number of datasets
        n_figs = (n_total + data_per_fig - 1) // data_per_fig  # Number of figures needed

        # Plot multiple figures
        for fig_idx in range(n_figs):
            fig, ax = plt.subplots()
            
            # Determine the range of data to plot in this figure
            start_idx = fig_idx * data_per_fig
            end_idx = min((fig_idx + 1) * data_per_fig, n_total)
            
            for i in range(start_idx, end_idx):
                ax.plot(ts, sds[:, i], label=ns[i])
            ax.plot(ts, np.zeros_like(sds[:, 0]), 'r--')
            ax.set_title(f"Collision Distance (Pinocchio) - Figure {fig_idx + 1}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Sd(q) (m)")
            ax.grid('on')
            ax.legend()


    def plot_cmds(self, axes=None, index=0, legend=None):
        ts = self.data["ts"]
        cmd_vels = self.data["cmd_vels"]
        cmd_accs = self.data.get("cmd_accs")
        # cmd_jerks = self.data.get("cmd_jerks")
        ref_vels = self.data.get("ref_vels", [])
        ref_accs = self.data.get("ref_accs", [])
        nq = int(self.data["nq"])
        nv = int(self.data["nv"])

        if axes is None:
            axes = []
            for i in range(2):
                f, ax = plt.subplots(nv, 1, sharex=True, figsize=(13, 23))
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
                label=legend + f"$v_{{cmd_{i + 1}}}$" + f"max = " + str(max(cmd_vels[:, i])),
                color=colors[index],
            )
            if len(ref_vels) > 0:
                ax[i].plot(
                    ts[:len(ref_vels)],
                    ref_vels[:, i],
                    label=legend + f"$v_{{plan_{i + 1}}}$",
                    linestyle="--",
                    color='r',
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
                    label=legend + f"$a_{{cmd_{i + 1}}}$" + f"max = " + str(max(cmd_accs[:, i])),
                    color=colors[index],
                )
                if len(ref_accs) > 0:
                    ax[i].plot(
                        ts[:len(ref_accs)],
                        ref_accs[:, i],
                        label=legend + f"$a_{{plan_{i + 1}}}$",
                        linestyle="--",
                        color='r',
                    )

                ax[i].grid()
                ax[i].legend()
            ax[-1].set_xlabel("Time (s)")
            ax[0].set_title("Commanded joint acceleration (rad/s^2)")

        # if cmd_jerks is not None:
        #     ax = axes[2]
        #     for i in range(nv):
        #         ax[i].plot(
        #             ts[:-1],
        #             cmd_jerks[:, i],
        #             '-x',
        #             label=legend + f"$j_{{cmd_{i + 1}}}$" + f"max = " + str(self.data["statistics"]["cmd_jerks"]["max"][i]),
        #             linestyle="--",
        #             color=colors[index],
        #         )

        #         ax[i].grid()
        #         ax[i].legend()
        #     ax[-1].set_xlabel("Time (s)")
        #     ax[0].set_title("Commanded joint jerk (rad/s^3)")

        return axes

    def plot_cmds_normalized(self, axes=None, index=0, legend=None):
        ts = self.data["ts"]
        cmd_vels = self.data["cmd_vels_normalized"]
        cmd_accs = self.data.get("cmd_accs_normalized")
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
                color=colors[index],
            )

            ax[i].grid()
            ax[i].legend()
        ax[-1].set_xlabel("Time (s)")
        ax[0].set_title("Commanded joint velocity Normalized")

        ax = axes[1]
        for i in range(nv):
            ax[i].plot(
                ts,
                cmd_accs[:, i],
                '-x',
                label=legend + f"$a_{{cmd_{i + 1}}}$",
                linestyle="--",
                color=colors[index],
            )

            ax[i].grid()
            ax[i].legend()
        ax[-1].set_xlabel("Time (s)")
        ax[0].set_title("Commanded joint acceleration Normalized")

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
        ref_vels = self.data.get("ref_vels", [])
        nq = int(self.data["nq"])
        nv = int(self.data["nv"])

        if axes is None:
            axes = []
            f, axes = plt.subplots(nv, 1, sharex=True)

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        for i in range(nv):
            axes[i].plot(ts, xs[:, nq + i],'.', label=f"$v_{i+1}$", color=colors[i])
            axes[i].plot(
                ts,
                cmd_vels[:, i], '-x',
                label=f"$v_{{cmd_{i + 1}}}$",
                # linestyle="-x",
                color=colors[i],
            )
            if len(ref_vels) > 0:
                axes[i].plot(
                    ts[:len(ref_vels)],
                    ref_vels[:, i], '-o',
                    label=f"$v_{{plan_{i + 1}}}$",
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

    def plot_state_normalized(self):
        self.plot_value_vs_time(
            "xs_normalized",
            indices=range(self.data["nq"]),
            legend_prefix="q",
            ylabel="Joint Position Normalized (range: [-1, 1])",
            title="Joint Positions Normalized vs. Time",
        )
        self.plot_value_vs_time(
            "xs_normalized",
            indices=range(self.data["nq"], self.data["nq"] + self.data["nv"]),
            legend_prefix="v",
            ylabel="Joint Velocity Normalized (range: [-1, 1])",
            title="Joint Velocities Normalized vs. Time",
        )

    def plot_run_time(self, axes=None, index=0, block=True, legend=None):
        # Time x HT-Iter x Task x ST-ITer+1
        t_sim = self.data["ts"]
        run_time = self.data.get("controller_run_time")
        if run_time is None:
            print("Ignore run time")
            return

        if legend is None:
            legend = self.data["name"]
        print(self.data["name"])

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

    def plot_cost_separate(self, axes=None, index=0, block=True, legend=None):
                    
        if legend is None:
            legend = self.data["name"]
        cost_num = len(self.controller.cost)
        if axes is None:
            axes = []
            for k in range(cost_num):
                f, ax = plt.subplots()
                axes.append(ax)

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

        if cost is None and cost_final is None:
            print("htmpc ignored. mpc_cost_iters and mpc_cost_finals found")
            return 
        
        if cost is not None:
            if len(cost.shape) == 4:
                N, HT_iter_num, task_num, ST_iter_num = cost.shape
            elif len(cost.shape) == 3:
                N, task_num, ST_iter_num = cost.shape
                HT_iter_num = 1
                cost = cost.reshape(N, HT_iter_num, task_num, ST_iter_num)
                cost_final = cost_final.reshape(N, task_num)

            elif len(cost.shape) == 1:
                N = len(cost)
                task_num, ST_iter_num, HT_iter_num = 1,1,1
                cost = cost.reshape(N, HT_iter_num, task_num, ST_iter_num)
                cost_final = cost_final.reshape(N, task_num)
        else:
            if len(cost_final.shape) == 2:
                N, task_num = cost_final.shape
            else:
                N = len(cost_final)
                task_num = 1
                cost_final = cost_final.reshape(N, task_num)
            
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

        if cost is not None:
            xticks = np.arange(N) * (HT_iter_num * ST_iter_num)
            xlables = [str(t) for t in t_sim]
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

        if cost_final is not None:
            for l in range(task_num):
                ax[l].plot(t_sim, cost_final[:, l], ".-", label=legend, linewidth=3, markersize=10)
                ax[l].set_title("Task " + str(l) + " Final Cost", fontsize=20)

            ax[0].legend(fontsize=20)
        plt.show(block=block)
        return axes

    def plot_solver_status_htmpc(self, axes=None, index=0, block=True, legend=None):
        axes = self.plot_time_series_data_htmpc("mpc_solver_statuss", axes, index, False, legend)
        self.plot_time_series_data_htmpc("mpc_step_sizes", axes, index, block, legend)

        return axes

    def plot_solver_iters_htmpc(self, axes=None, index=0, block=True, legend=None):

        axes = self.plot_time_series_data_htmpc("mpc_qp_iters", axes, index, False, legend)
        self.plot_time_series_data_htmpc("mpc_sqp_iters", axes, index, block, legend)

        return axes
    
    def plot_time_htmpc(self, axes=None, index=0, block=True, legend=None):
        
        for data_name in self.data.keys():
            if "mpc_time" in data_name:
                axes = self.plot_time_series_data_htmpc(data_name, axes, index, False, legend)

        return axes

    def plot_time_series_data_htmpc(self, data_name, axes=None, index=0, block=True, legend=None):
        # Time x HT-Iter x Task x ST-ITer
        data = self.data.get(data_name)
        if data is None:
            print(f"Did not find {data_name}. Stop plotting.")
            return
        
        if len(data.shape) == 4:
            N, HT_iter_num, task_num, ST_iter_num = data.shape
        elif len(data.shape) == 2:
            N, task_num = data.shape
            HT_iter_num, ST_iter_num = 1, 1
            data = data.reshape(N, HT_iter_num, task_num, ST_iter_num)
        elif len(data.shape) == 1:
            N = len(data)
            task_num, ST_iter_num, HT_iter_num = 1,1,1
            data = data.reshape(N, HT_iter_num, task_num, ST_iter_num)

        t_sim = self.data["ts"]
        if legend is None:
            legend = self.data["name"]

        if axes is None:
            f, axes = plt.subplots(task_num, 1, sharex=True)
            if task_num == 1:
                axes = [axes]
        if HT_iter_num == task_num == ST_iter_num == 1:
            axes[0].plot(t_sim, data.squeeze(), ".-", label=" ".join([legend]+ data_name.split("_")[1:]), linewidth=2, markersize=8)
            axes[0].grid('on')
        
        else:
            xticks = np.arange(N) * (HT_iter_num * ST_iter_num)
            xlables = [str(t) for t in t_sim]
            for l in range(task_num):
                axes[l].plot(data[:, :, l, :].flatten(), "x-", label=" ".join([legend]+ data_name.split("_")[1:]), linewidth=2, markersize=8)
                axes[l].set_title("Task" + str(l))
                axes[l].set_xticks(xticks)
                axes[l].set_xticklabels(xlables)
                axes[l].grid('on')
        axes[-1].set_xlabel("Time (s)")
        axes[0].legend()
        plt.show(block=block)

        return axes
    
    def plot_task_performance(self, axes=None, index=0, legend=None):
        if axes is None:
            f, axes = plt.subplots(4, 1, sharex=True)
        else:
            if len(axes) != 4:
                raise ValueError("Given axes number ({}) does not match task number ({}).".format(len(axes), 4))

        if legend is None:
            legend = self.data["name"]

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        t_sim = self.data["ts"]
        nq = int(self.data["nq"])
        xs_sat, us_sat = self.model_interface.robot.checkBounds(self.data["xs"], self.data["cmd_accs"], -1e-3)
        # axes[0].plot(t_sim, xs_sat + us_sat, label=legend + " " + "mean = " + str(np.mean(xs_sat + us_sat)), color=colors[index])
        axes[0].plot(t_sim, self.data["constraints_violation"]*100, label=legend + " mean = {:.3f}".format(self.data["statistics"]["constraints_violation"]["mean"]*100), color=colors[index])
        axes[0].set_ylabel("constraints violation (%)")

        # axes[1].plot(t_sim, self.data["err_ee"],
        #              label=legend + " sum = {:.3f} RMS = {:.3f}".format(self.data["statistics"]["err_ee"]["integral"],
        #                                                                 self.data["statistics"]["err_ee"]["rms"]), color=colors[index])
        axes[1].plot(t_sim, self.data["err_ee"],
                     label=legend + " acc = {:.3f}".format(self.data["statistics"]["err_ee"]["integral"]), color=colors[index])
        axes[1].set_ylabel("EE Err (m)")

        # axes[2].plot(t_sim, self.data["err_base"],
        #              label=legend + " sum = {:.3f} RMS = {:.3f}".format(self.data["statistics"]["err_base"]["integral"],
        #                                                                 self.data["statistics"]["err_base"]["rms"]),
        #              color=colors[index])
        axes[2].plot(t_sim, self.data["err_base"],
                     label=legend + " acc = {:.3f}".format(self.data["statistics"]["err_base"]["integral"]), color=colors[index])
        axes[2].set_ylabel("Base Err (m)")

        axes[3].plot(t_sim, self.data["arm_manipulability"], label=legend)
        axes[3].set_ylabel("Arm Manipulability")
        axes[3].set_xlabel("Time (s)")

        for a in axes:
            a.legend()

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
            f, axes = plt.subplots(task_num+1, 1, sharex=True)
        else:
            if len(axes) != task_num+1:
                raise ValueError("Given axes number ({}) does not match task number ({}).".format(len(axes), task_num))

        if legend is None:
            legend = self.data['name']

        t_sim = self.data["ts"]
        N = len(t_sim)
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        for tid in range(task_num):
            w = ws[tid]

            # if tid == 0:
            if False:
                saturation_num = np.sum(w > 1e-3, axis=1)
                axes[tid].plot(t_sim, saturation_num, color=colors[index], label=legend)
                axes[tid].set_ylabel("# Inequalities Saturated \n out of {}".format(w.shape[1]))
            else:
                axes[tid].plot(t_sim, np.linalg.norm(w, axis=1), color=colors[index], label=legend)

            axes[tid].set_title("Task " + str(tid) + " " + task_names[tid])

        axes[-1].plot(t_sim, self.data["arm_manipulability"], color=colors[index], label=legend )
        axes[-1].set_title("Arm Manipulability")
        axes[-1].set_xlabel("Time (s)")
        plt.legend()

        plt.figure()
        w0 = ws[0]
        plt.plot(t_sim, np.linalg.norm(w0[:, :18], axis=1),label=legend + "_joint_vel")
        plt.plot(t_sim, np.linalg.norm(w0[:, 18:36], axis=1),label=legend + "_joint_angle")
        plt.plot(t_sim, np.linalg.norm(w0[:, 36:54], axis=1),label=legend + "_joint_acc")
        plt.plot(t_sim, np.linalg.norm(w0[:, 54:], axis=1),label=legend + "_collision")
        plt.legend()

        return axes

    def plot_mpc_prediction(self, data_name, t=0, axes=None, index=0, block=True, legend=None, p_struct=None):
        t_sim = self.data["ts"]
        t_index = np.argmin(np.abs(t_sim - t))
        off_set = t_index % 4
        # Time x prediction step x data dim
        data = self.data.get(data_name, None)
        if data is None and "constraint" in data_name.split("_"):
            # mpc constraints with prediction
            N = len(self.data["ts"])
            data_name_split = data_name.split("_")
            name_end_index = data_name_split.index("constraint")
            constraint_name = "_".join(data_name_split[1:name_end_index])
            for constraint in self.controller.constraints:
                print("checking {} against {}".format(constraint_name,constraint.name))

                if constraint.name !=constraint_name:
                    continue
                self.data[data_name] = []
                for t_index in range(N):
                    if p_struct is None:
                        param_map_bar = [reconstruct_sym_struct_map_from_array(self.controller.p_struct, param_map_array) for param_map_array in self.data["mpc_ocp_params"][t_index]]
                    else:
                        param_map_bar = [reconstruct_sym_struct_map_from_array(p_struct, param_map_array) for param_map_array in self.data["mpc_ocp_params"][t_index]]

                    x_bar = self.data["mpc_x_bars"][t_index]
                    u_bar = self.data["mpc_u_bars"][t_index]
                    self.data[data_name].append(self.controller.evaluate_constraints(constraint, x_bar, u_bar, param_map_bar))
                self.data[data_name] = np.array(self.data[data_name])
            if self.data.get(data_name, None) is None:
                return
        print(data_name)
        data = self.data.get(data_name, None)
        print(data.shape)
        
        if len(data.shape) ==4:
            data = data.squeeze(axis=-1)

        print(data.shape)
        if len(data.shape) == 3:
            N, P, D = data.shape
        
        data_per_figure = 6
        if D%data_per_figure == 0:
            figs_num = D//data_per_figure
        else:
            figs_num = D//data_per_figure + 1



        if legend is None:
            legend = self.data["name"]

        t_sim = self.data["ts"]
        mpc_dt = self.config["controller"]["dt"]
        t_prediction = np.arange(P) * mpc_dt
        t_all = t_sim.reshape((N, 1)) + t_prediction.reshape((1, P))

        for fi in range(figs_num):
            if axes is None or fi!=0:
                data_num = data_per_figure if fi < figs_num - 1 else D - fi*data_per_figure
                f, axes = plt.subplots(data_num, 1, sharex=True)
                if data_num ==1:
                    axes=[axes]
            else:
                data_num = len(axes)

            for i in range(data_num):
                data_index = i+fi*data_per_figure
                axes[i].plot(t_all[:, 0].flatten(), data[:,0, data_index].flatten(), "o-", label=" ".join(["actual", legend]), linewidth=2, markersize=8)
                axes[i].plot(t_all[off_set::5].T, data[off_set::5, :, data_index].T, "o-", linewidth=1, fillstyle='none')
                
                axes[i].set_ylabel("d[{}]".format(data_index))

                axes[i].legend()
                axes[i].grid()
            axes[0].set_title(" ".join(data_name.split("_")[1:] + ["figure", str(fi+1)+"/"+str(figs_num)]))
            
        plt.show(block=block)
    
        return axes

    def plot_mpc_prediction_cbf_gamma(self, data_name, t=0, axes=None, index=0, block=True, legend=None):
        t_sim = self.data["ts"]
        t_index = np.argmin(np.abs(t_sim - t))
        off_set = t_index % 4
        # Time x prediction step x data dim
        # mpc constraints with prediction
        N = len(self.data["ts"])
        data_name_split = data_name.split("_")
        name_end_index = data_name_split.index("constraint")
        constraint_name = "_".join(data_name_split[1:name_end_index])
        for constraint in self.controller.constraints:
            print("checking {} against {}".format(constraint_name,constraint.name))

            if constraint.name !=constraint_name:
                continue

            data_name = "mpc_sdf_gamma_constraint_prediction"
            self.data[data_name] = []
            for t_index in range(N):
                param_map_bar = [reconstruct_sym_struct_map_from_array(self.controller.p_struct, param_map_array) for param_map_array in self.data["mpc_ocp_params"][t_index]]
                x_bar = self.data["mpc_x_bars"][t_index]
                u_bar = self.data["mpc_u_bars"][t_index]
                self.data[data_name].append(self.controller.evaluate_cbf_gammas(constraint, x_bar, u_bar, param_map_bar))
            self.data[data_name] = np.array(self.data[data_name])
        if self.data.get(data_name, None) is None:
            return
        data = self.data.get(data_name, None)
        
        if len(data.shape) ==4:
            data = data.squeeze(axis=-1)

        if len(data.shape) == 3:
            N, P, D = data.shape
        
        data_per_figure = 6
        if D%data_per_figure == 0:
            figs_num = D//data_per_figure
        else:
            figs_num = D//data_per_figure + 1



        if legend is None:
            legend = self.data["name"]

        t_sim = self.data["ts"]
        mpc_dt = self.config["controller"]["dt"]
        t_prediction = np.arange(P) * mpc_dt
        t_all = t_sim.reshape((N, 1)) + t_prediction.reshape((1, P))

        for fi in range(figs_num):
            if axes is None or fi!=0:
                data_num = data_per_figure if fi < figs_num - 1 else D - fi*data_per_figure
                f, axes = plt.subplots(data_num, 1, sharex=True)
                if data_num ==1:
                    axes=[axes]
            else:
                data_num = len(axes)

            for i in range(data_num):
                data_index = i+fi*data_per_figure
                axes[i].plot(t_all[:, 0].flatten(), data[:,0, data_index].flatten(), "o-", label=" ".join(["actual", legend]), linewidth=2, markersize=8)
                axes[i].plot(t_all[off_set::4].T, data[off_set::4, :, data_index].T, "o-", linewidth=1, fillstyle='none')
                
                axes[i].set_ylabel("d[{}]".format(data_index))

                axes[i].legend()
                axes[i].grid()
            axes[0].set_title(" ".join(data_name.split("_")[1:] + ["figure", str(fi+1)+"/"+str(figs_num)]))
            
        plt.show(block=block)
    
        return axes

    def show(self):
        plt.show()

    def plot_all(self):
        self.data["name"] = ""
        self.plot_ee_tracking()
        self.plot_ee_linear_velocity_tracking()
        self.plot_base_tracking()
        self.plot_base_velocity_tracking()
        self.plot_tracking_err()
        self.plot_cmd_vs_real_vel()
        self.plot_state()

        self.plot_cost_htmpc()
        self.plot_solver_status_htmpc()
        self.plot_solver_iters_htmpc()
        self.plot_run_time()

        self.save_figs()

    def save_figs(self):

        figs = [plt.figure(n) for n in plt.get_fignums()]
        folder_name = str(self.data["dir_path"]).split("/")[-1]

        multipage(Path(str(self.data["folder_path"]))/Path(folder_name) / "report.pdf", figs)

    def plot_mpc(self):
        self.plot_cost_htmpc(block=False)
        self.plot_solver_status_htmpc(block=False)
        self.plot_solver_iters_htmpc(block=False)
        self.plot_time_htmpc(block=False)
        # self.plot_solver_iters()
        self.plot_time_series_data_htmpc("time_get_map", block=False)
        self.plot_mpc_prediction("mpc_sdf_constraint_predictions",t=0.1, block=False)
        self.plot_mpc_prediction("mpc_self_constraint_predictions", block=False)
        # self.plot_mpc_prediction("mpc_EEPos3_Lex_constraint_predictions", block=False)


        # self.plot_solver_iters(block=False)
        # self.plot_mpc_prediction("mpc_obstacle_cylinder_1_link_constraints")
        # self.plot_mpc_prediction("mpc_control_constraint_predictions",block=False)
        # self.plot_mpc_prediction("mpc_state_constraint_predictions", block=False)
        self.plot_mpc_prediction("mpc_x_bars",t=0.1, block=False)
        self.plot_mpc_prediction("mpc_u_bars",t=0.1, block=False)

        self.plot_run_time()

        mpc_failure_steps = np.where(self.data["mpc_solver_statuss"] == 4)[0]
        if mpc_failure_steps.size > 0:
            self.mpc_constraint_debug(self.data["ts"][mpc_failure_steps[0]])

    def plot_robot(self):
        self.plot_cmd_vs_real_vel()
        self.plot_state()
        self.plot_state_normalized()
        self.plot_cmds()
        # self.plot_cmds_normalized()
        self.plot_du()
        self.plot_collision()
        self.plot_collision_detailed()

    def plot_tracking(self):
        self.plot_tracking_err()
        # self.plot_cmd_vs_real_vel()
        self.plot_task_performance()
        self.plot_task_violation()

        self.plot_ee_tracking(base_frame=False)
        self.plot_ee_orientation_tracking()
        self.plot_ee_linear_velocity_tracking()
        self.plot_base_tracking()
        self.plot_base_velocity_tracking()
        axes = self.plot_base_path()
        self.plot_base_ref_path(axes)


    def plot_quick_check(self):
        self.plot_task_performance()
        self.plot_collision()
    
    def plot_time_optimal_plan_tracking_results(self):
        self.plot_tracking()
        self.plot_cmds()

        self.plot_state_normalized()
        self.plot_cmds_normalized()
        self.plot_collision()
    
    def plot_exp(self, time):
        self.plot_ee_tracking()
        self.plot_sdf_map(t=time)
        self.plot_collision()
        self.plot_time_htmpc(block=False)
        self.plot_mpc_prediction("mpc_sdf_constraint_predictions",t=0, block=False)
        self.plot_mpc_prediction_cbf_gamma("mpc_sdf_constraint_predictions",t=0, block=False)

    
    def plot_sdf(self, t, use_iter_snapshot=False, block=True):
        t_sim = self.data["ts"]
        t_index = np.argmin(np.abs(t_sim - t))
        if use_iter_snapshot:
            iter_snapshot = dict(enumerate(self.data["mpc_iter_snapshots"][t_index].flatten(), 1))[1]
            param_map_array = iter_snapshot["p_map_bar"][0]
            param_map = reconstruct_sym_struct_map_from_array(self.controller.p_struct, param_map_array)
            x_bar = np.array(iter_snapshot["x_bar"])
            u_bar = np.array(iter_snapshot["u_bar"])
        else:
            param_map_array = self.data["mpc_ocp_params"][t_index][0]
            param_map = reconstruct_sym_struct_map_from_array(self.controller.p_struct, param_map_array)
            x_bar = self.data["mpc_x_bars"][t_index]
            u_bar = self.data["mpc_u_bars"][t_index]
        sdf_map = self.model_interface.sdf_map

        # for i in range(len(self.data["mpc_ocp_params"][t_index])-1):
        #     param_map = self.data["mpc_ocp_params"][t_index][i]
        #     param_next = self.data["mpc_ocp_params"][t_index][i+1]
        #     keys = ['x_grid_sdf', 'y_grid_sdf', 'z_grid_sdf', 'value_sdf']
        #     for key in keys:
        #         diff = np.linalg.norm(param_map[key].toarray().flatten() - param_next[key].toarray().flatten())
        #         print(f"{key} diff {diff}")
        sdf_map.update_map(param_map["x_grid_sdf"].toarray().flatten(),
                           param_map["y_grid_sdf"].toarray().flatten(),
                           param_map["z_grid_sdf"].toarray().flatten(),
                           param_map["value_sdf"].toarray().flatten())
        
        ee_pos = self.data["mpc_ee_poss"][t_index]
        sd_val = sdf_map.query_val(ee_pos[:, 0],ee_pos[:, 1],ee_pos[:, 2]).flatten()
        sd_grad = sdf_map.query_grad(ee_pos[:, 0],ee_pos[:, 1],ee_pos[:, 2])
        sd_grad = sd_grad.reshape((3,-1))

        # sample around the trajectory
        # sample_grid = []
        # for i in range(3):
        #     sample_grid.append(np.linspace(min(ee_pos[:, i]), max(ee_pos[:, i]), 8))
        # sample_mesh = np.meshgrid(*sample_grid)
        # sample_mesh = [g.flatten() for g in sample_mesh]
        # sd_val_sample = sdf_map.query_val(*sample_mesh)
        # sd_grad_sample = sdf_map.query_grad(*sample_mesh)
        # sd_grad_sample = sd_grad_sample.reshape((3,-1))

        const = SignedDistanceConstraintCBF(self.model_interface.robot, self.model_interface.getSignedDistanceSymMdls("sdf"), 
                                            self.config["controller"]["collision_safety_margin"]["sdf"], "sdf")
        cst_p_dict = const.get_p_dict()
        cst_p_struct = casadi_sym_struct(cst_p_dict)
        cst_param_map = cst_p_struct(0)

        cst_param_map["x_grid_sdf"] = param_map["x_grid_sdf"]
        cst_param_map["y_grid_sdf"] = param_map["y_grid_sdf"]
        cst_param_map["z_grid_sdf"] = param_map["z_grid_sdf"]
        cst_param_map["value_sdf"] = param_map["value_sdf"]
        cst_param_map["gamma_sdf"] = param_map["gamma_sdf"]

        g_sdf = const.g_fcn(x_bar[:-1].T, u_bar.T, cst_param_map.cat).toarray()
        print(f"g_sdf{g_sdf}")
        # h_sdf = const.h_fcn(x_bar[:-1].T, u_bar.T, cst_param_map.cat).toarray()
        # print(f"h_sdf{h_sdf}")


        # sample around one point
        violated_constraints_indices = np.where(g_sdf>1e-3)
        if len(violated_constraints_indices[0])>0:
            prediction_index = violated_constraints_indices[1][0]
        else:
            print("No violation found")
            prediction_index = 0
        # prediction_index = 5
        print(prediction_index)
        sample_center = ee_pos[prediction_index]
        x_grid = np.linspace(-0.025, 0.025, 5) + sample_center[0]
        y_grid = np.linspace(-0.025, 0.025, 5) + sample_center[1]
        z_grid = np.linspace(0, 0, 1) + sample_center[2]
        sample_grid = [x_grid, y_grid, z_grid]
        sample_mesh = np.meshgrid(*sample_grid)
        sample_mesh = [g.flatten() for g in sample_mesh]
        sd_val_sample = sdf_map.query_val(*sample_mesh)
        sd_grad_sample = sdf_map.query_grad(*sample_mesh)
        sd_grad_sample = sd_grad_sample.reshape((3,-1))
        print(sd_grad_sample)

        # xdot = const.xdot_fcn(x_bar[prediction_index].T, u_bar[prediction_index].T, cst_param_map.cat).toarray()
        # print(f"xdot{xdot}")
        # print(f"x_bar : {x_bar}")
        # print(f"u_bar : {u_bar}")
        # h_grad = const.h_grad_fcn(x_bar[prediction_index] + np.array([0,+0.0]+16*[0]), u_bar[1], cst_param_map.cat).toarray()
        # print(f"h_grad{h_grad}")
        # print(h_grad@xdot)
        # h_hess_fd= const.h_hess_fd_fcn(x_bar[prediction_index], u_bar[1], cst_param_map.cat).toarray()
        # print(f"h_hess_fd: {h_hess_fd}")
        # print(f"tr(h_hess_fd): {np.trace(np.abs(h_hess_fd))}")
        # g_hess_fd= const.g_hess_fd_fcn(x_bar[prediction_index], u_bar[1], cst_param_map.cat).toarray()
        # print(f"g_hess_fd: {g_hess_fd}")
        # print(f"tr(g_hess_fd): {np.trace(g_hess_fd)}")


        safety_margin = self.config["controller"]["collision_safety_margin"]["sdf"]
        r = 0.26
        collision_free = np.where(sd_val-r - safety_margin>=0)[0]
        collided = np.where(sd_val-r - safety_margin<0)[0]
        print(f"sd_val{sd_val-r - safety_margin}")
        print(f"collision_free index {collision_free}")

        print(np.linalg.norm(sd_grad, axis=0))
        fig1= plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.plot(ee_pos[collision_free, 0],ee_pos[collision_free, 1],ee_pos[collision_free, 2], color='g')
        ax1.plot(ee_pos[collided, 0],ee_pos[collided, 1],ee_pos[collided, 2], color='r')

        ax1.quiver(ee_pos[:, 0],ee_pos[:, 1],ee_pos[:, 2],
                   sd_grad[0]/5,sd_grad[1]/5, sd_grad[2]/5, arrow_length_ratio=0.2, normalize=False)
        ax1.quiver(sample_mesh[0],sample_mesh[1], sample_mesh[2],
                   sd_grad_sample[0]/5,sd_grad_sample[1]/5, sd_grad_sample[2]/5, arrow_length_ratio=0.2, normalize=False, color='r')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.set_title(f"Time: {t_sim[t_index]}s")
        # ax1.set_aspect('equal')

        x_lim = [-2, 2]
        y_lim = [-1, 2]
        z_lim = [0.18, 0.43]
        sdf_map.vis(x_lim=x_lim,
                    y_lim=y_lim,
                    z_lim=[0.1, 0.1],
                    block=False)
        sdf_map.vis(x_lim=x_lim,
            y_lim=y_lim,
            z_lim=[0.2, 0.2],
            block=False)
        sdf_map.vis(x_lim=x_lim,
            y_lim=y_lim,
            z_lim=[0.3, 0.3],
            block=False)
        sdf_map.vis(x_lim=x_lim,
            y_lim=y_lim,
            z_lim=[0.4, 0.4],
            block=False)
        
        # g_hess_fd = []
        # for i in range(u_bar.shape[0]):
        #     g_hess_fd.append(const.g_hess_fd_fcn(x_bar[i], u_bar[i], cst_param_map.cat).toarray())
        
        # tr_g_hess_fd = [np.trace(g) for g in g_hess_fd]

        # fig2= plt.figure()
        # plt.plot(tr_g_hess_fd, '-x')
        # plt.plot(prediction_index, tr_g_hess_fd[prediction_index], 'r-x')
        # plt.xlabel("MPC prediction step")
        # plt.ylabel("tr(g_cbf_hess)")

        plt.show(block=block)

    def plot_sdf_map(self, t, use_iter_snapshot=False, block=True, p_struct=None):
        t_sim = self.data["ts"]
        t_index = np.argmin(np.abs(t_sim - t))
        if use_iter_snapshot:
            iter_snapshot = dict(enumerate(self.data["mpc_iter_snapshots"][t_index].flatten(), 1))[1]
            param_map_array = iter_snapshot["p_map_bar"][0]
            param_map = reconstruct_sym_struct_map_from_array(self.controller.p_struct, param_map_array)
            x_bar = np.array(iter_snapshot["x_bar"])
            u_bar = np.array(iter_snapshot["u_bar"])
        else:
            param_map_array = self.data["mpc_ocp_params"][t_index][0]
            if p_struct is None:
                param_map = reconstruct_sym_struct_map_from_array(self.controller.p_struct, param_map_array)
            else:
                param_map = reconstruct_sym_struct_map_from_array(p_struct, param_map_array)

            x_bar = self.data["mpc_x_bars"][t_index]
            u_bar = self.data["mpc_u_bars"][t_index]
        sdf_map = self.model_interface.sdf_map
        sdf_map.update_map(param_map["x_grid_sdf"].toarray().flatten(),
                           param_map["y_grid_sdf"].toarray().flatten(),
                           param_map["z_grid_sdf"].toarray().flatten(),
                           param_map["value_sdf"].toarray().flatten())
        
        x_lim = [np.min(param_map["x_grid_sdf"]), np.max(param_map["x_grid_sdf"])]
        y_lim = [np.min(param_map["y_grid_sdf"]), np.max(param_map["y_grid_sdf"])]
        z_lim = [np.min(param_map["z_grid_sdf"]), np.max(param_map["z_grid_sdf"])]

        z_lims = np.linspace(z_lim[0], z_lim[1], int((z_lim[1] - z_lim[0])/0.1))
        z_lims = [0.1, 0.7]
        print("Visualizing Map at Time {}".format(t_sim[t_index]))
        for z in z_lims:
            sdf_map.vis(x_lim=x_lim,
                        y_lim=y_lim,
                        z_lim=[z, z],
                        block=False)


    def check_sdf(self):
        t_sim = self.data["ts"]
        N = len(t_sim)

        for i in range(N-1):
            param_map_array = self.data["mpc_ocp_params"][i][0]
            param_map = reconstruct_sym_struct_map_from_array(self.controller.p_struct, param_map_array)
            param_map_array_next = self.data["mpc_ocp_params"][i+1][0]
            param_map_next = reconstruct_sym_struct_map_from_array(self.controller.p_struct, param_map_array_next)
            keys = ['x_grid_sdf', 'y_grid_sdf', 'z_grid_sdf', 'value_sdf']
            for key in keys:
                diff = np.linalg.norm(param_map[key].toarray().flatten() - param_map_next[key].toarray().flatten())
                print(f"time{t_sim[i]} {key} diff {diff}")
    
    def sdf_failure_debug(self, t):
        f, axes = plt.subplots(2, 1, sharex=True)
        self.plot_mpc_prediction("mpc_sdf_constraints", t, axes=[axes[0]], block=False)
        self.plot_time_series_data_htmpc("mpc_solver_statuss", axes=[axes[1]], block=False)
        self.plot_sdf(t, use_iter_snapshot=True, block=True)

    def mpc_constraint_debug(self, t):
        self.plot_time_series_data_htmpc("mpc_solver_statuss", block=False)

        for constraint in self.controller.constraints:
            data_name = "_".join(["mpc", constraint.name, "constraint", "predictions"])
            self.plot_mpc_prediction(data_name, t, block=False)

        # self.plot_mpc_prediction("mpc_sdf_constraints", t, axes=[axes[0]], block=False)
        # self.plot_mpc_prediction("mpc_control_constraints", t,block=False)
        # self.plot_mpc_prediction("mpc_state_constraints", t, block=False)
            
    def print_mpc_param(self, param_name, p_struct=None):
        if p_struct is None:
            param_map_bar = [reconstruct_sym_struct_map_from_array(self.controller.p_struct, param_map_array[0]) for param_map_array in self.data["mpc_ocp_params"]]
        else:
            param_map_bar = [reconstruct_sym_struct_map_from_array(p_struct, param_map_array[0]) for param_map_array in self.data["mpc_ocp_params"]]
        
        if param_name in param_map_bar[0].keys():
            print("Param {}".format(param_name))
            for k, param in enumerate(param_map_bar):
                print("time: {} param:{}".format(self.data["ts"][k], param[param_name]))
        else:
            print("Param {} Not Found".format(param_name))

class ROSBagPlotter:
    def __init__(self, bag_file, config_file="$(rospack find mmseq_run)/config/robot/thing.yaml"):
        self.data = {"ur10": {}, "ridgeback": {}, "mpc": {}, "vicon": {}, "model":{}}
        self.bag = rosbag.Bag(bag_file)   
        self.config = load_config(config_file)
        self.robot = MobileManipulator3D(self.config["controller"])

        self.parse_joint_states(self.bag)
        self.parse_cmd_vels(self.bag)
        self.parse_mpc_tracking_pt(self.bag)
        self.parse_vicon_msgs(self.bag)
        self.parse_odom(self.bag)
        self.compute_values_from_robot_model()
        # self._set_zero_time()
        self.parse_base_tf_msgs(self.bag)

    def parse_base_tf_msgs(self, bag):
        tf_msgs = [msg for _, msg, _ in bag.read_messages("/tf")]
        ts, p_bws, orn_bws = ros_utils.parse_tf_messages(tf_msgs, "base_link", "my_world")
        orn_wbs = np.array([math.quat_inverse(q) for q in orn_bws])
        p_wbs = np.array([math.quat_rotate(q, -p) for p, q in zip(p_bws, orn_wbs)])
        rpys = np.array([euler_from_quaternion(q) for q in orn_wbs])

        if len(ts) > 0:
            valid_indx = [0]
            t_new = [ts[0]]
            for i, t in enumerate(ts):
                if t > t_new[-1]:
                    t_new.append(t)
                    valid_indx.append(i)
            valid_indx = np.array(valid_indx)
            t_new = np.array(t_new)
            self.data["tf"] = {"base_link": {"ts": t_new,
                                            "pos": p_wbs[valid_indx],
                                            "orn": orn_wbs[valid_indx],
                                            "rpy": rpys[valid_indx]}}
            dts = t_new[1:] - t_new[:-1]
            pos = self.data["tf"]["base_link"]["pos"]
            print(ts)
            pos_diff = (pos[1:, :] - pos[:-1, :]) / np.expand_dims(dts, axis=-1)
            self.data["tf"]["base_link"]["pos_diff"] = pos_diff

            rpy = self.data["tf"]["base_link"]["rpy"]
            rpy_diff = (rpy[1:, :] - rpy[:-1, :]) / np.expand_dims(dts, axis=-1)
            self.data["tf"]["base_link"]["rpy_diff"] = rpy_diff
        else:
            self.data["tf"] = {"base_link": {"ts": np.zeros(0),
                                            "pos": np.zeros((0, 3)),
                                            "orn": np.zeros((0, 4)),
                                            "rpy": np.zeros((0, 3)),
                                            "pos_diff": np.zeros((0, 3)),
                                            "rpy_diff": np.zeros((0,3))}}

    def parse_joint_states(self, bag):

        ur10_msgs = [msg for _, msg, _ in bag.read_messages("/ur10/joint_states")]
        ridgeback_msgs = [msg for _, msg, _ in bag.read_messages("/ridgeback/joint_states")]

        tas, qas, vas = ros_utils.parse_ur10_joint_state_msgs(ur10_msgs, False)
        tbs, qbs, vbs = ros_utils.parse_ridgeback_joint_state_msgs(ridgeback_msgs, False)
        self.data["ur10"]["joint_states"] = {"ts": tas, "qs": qas, "vs": vas}           # 125hz
        self.data["ridgeback"]["joint_states"] = {"ts": tbs, "qs": qbs, "vs": vbs}      # 50hz
        
        dts = tbs[1:] - tbs[:-1]
        vel = (qbs[1:, :] - qbs[:-1, :]) / np.expand_dims(dts, axis=-1)
        self.data["ridgeback"]["joint_states"]["q_diffs"] = vel

        # Reconstruct body-frame ridgeback velocity
        vb_bs = [rotz(qbs[i, 2]).T @  vbs[i, :] for i in range(len(tbs))]
        self.data["ridgeback"]["joint_states"]["vbs"] = np.array(vb_bs)

        if qas.shape[0] != 0:
            fqa_interp = interp1d(tas, qas, axis=0, fill_value="extrapolate")
            fva_interp = interp1d(tas, vas, axis=0, fill_value="extrapolate")
            qas_interp = fqa_interp(tbs)
            vas_interp = fva_interp(tbs)
            self.data["ur10"]["joint_states_interpolated"] = {"ts": tbs, "qs": qas_interp, "vs": vas_interp}

        else:
            self.data["ur10"]["joint_states"] = {"ts": tas, "qs": np.zeros((0, 6)), "vs": np.zeros((0, 6))}
            self.data["ur10"]["joint_states_interpolated"] = {"ts": tas, "qs": np.zeros((0, 6)), "vs": np.zeros((0, 6))}

    def parse_odom(self, bag):
        odom_msgs = [msg for _, msg, _ in bag.read_messages("/odometry/filtered")]
        ts, vs, omegas = ros_utils.parse_ridgeback_odom_msgs(odom_msgs)
        self.data["ridgeback"]["odom"] = {}
        self.data["ridgeback"]["odom"]["vs"] = vs
        self.data["ridgeback"]["odom"]["omegas"] = omegas
        self.data["ridgeback"]["odom"]["ts"] = ts

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
            if name == THING_BASE_NAME:
                ts, qs = ros_utils.parse_ridgeback_vicon_msgs(msgs)         # q is a 3-element vector x, y, theta
                self.data["vicon"]["ThingBase"] = {"ts": ts, "pos": qs[:, :2], "orn": qs[:, 2]}

                dts = ts[1:] - ts[:-1]
                vel = (qs[1:, :2] - qs[:-1, :2]) / np.expand_dims(dts, axis=-1)
                self.data["vicon"]["ThingBase"]["pos_diff"] = vel

                yaw = self.data["vicon"]["ThingBase"]["orn"]
                yaw_diff = (yaw[1:] - yaw[:-1]) / dts
                self.data["vicon"]["ThingBase"]["yaw_diff"] = yaw_diff

            else:
                ts, qs = ros_utils.parse_transform_stamped_msgs(msgs, False)    # q is a 7-element vector x, y, z for position followed by a unit quaternion
                self.data["vicon"][name] = {"ts": ts, "pos": qs[:, :3], "orn": qs[:, 3:]}

    def compute_values_from_robot_model(self):
        f_base = self.robot.kinSymMdls[self.robot.base_link_name]
        f_ee = self.robot.kinSymMdls[self.robot.tool_link_name]
        J_ee = self.robot.jacSymMdls[self.robot.tool_link_name]

        r_base_s = []
        yaw_base_s = []
        v_base_s = []
        ω_base_s = []

        r_ee_s = []
        quat_ee_s = []
        v_ee_s = []
        ω_ee_s = []


        for i in range(len(self.data["ur10"]["joint_states_interpolated"]["ts"])):
            qa = self.data["ur10"]["joint_states_interpolated"]["qs"][i]
            qb = self.data["ridgeback"]["joint_states"]["qs"][i]
            q = np.hstack((qb, qa))

            q_dota = self.data["ur10"]["joint_states_interpolated"]["vs"][i]
            q_dotb = self.data["ridgeback"]["joint_states"]["vs"][i]
            q_dot = np.hstack((q_dotb, q_dota))
            r_b, theta_b = f_base(q)
            r_ee, rot_ee = f_ee(q)
            quat_ee = r2q(np.array(rot_ee), order="xyzs")
            v_ee = J_ee(q) @ q_dot

            r_base_s.append(r_b)
            yaw_base_s.append(theta_b)
            v_base_s.append(q_dotb[:2])
            ω_base_s.append(q_dotb[2])

            r_ee_s.append(r_ee)
            quat_ee_s.append(quat_ee)
            v_ee_s.append(v_ee)
            # TODO: compute angular velocity from model. Perhaps use a package that gives full jacobian casadi functino
            # Possible choice: https://pypi.org/project/cmeel-casadi-kin-dyn/
            ω_ee_s.append(np.zeros(3))

        self.data["model"]["EE"] = {"ts": self.data["ur10"]["joint_states_interpolated"]["ts"].copy(),
                                    "pos": np.array(r_ee_s).squeeze(),
                                    "vel_lin": np.array(v_ee_s).squeeze(),
                                    "orn": np.array(quat_ee_s).squeeze(),
                                    "vel_ang":np.array(ω_ee_s).squeeze()}

        self.data["model"]["base"] = {"ts": self.data["ur10"]["joint_states_interpolated"]["ts"].copy(),
                                      "pos": np.array(r_base_s).squeeze(),
                                      "vel_lin": np.array(v_base_s).squeeze(),
                                      "orn": np.array(yaw_base_s).flatten(),
                                      "vel_ang": np.array(ω_base_s).flatten()}

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
    
    def plot_base_odom_velocity(self):
        axes = []
        for i in range(2):
            f = plt.figure()
            axes.append(f.gca())

        axes[0].plot(self.data["ridgeback"]["odom"]["ts"], self.data["ridgeback"]["odom"]["vs"][:, 0], label="vx")
        axes[0].plot(self.data["ridgeback"]["odom"]["ts"], self.data["ridgeback"]["odom"]["vs"][:, 1], label="vy")
        axes[0].plot(self.data["ridgeback"]["odom"]["ts"], self.data["ridgeback"]["odom"]["vs"][:, 2], label="vz")
        axes[0].set_title("Ridgeback Odom Linear Velocity")

        axes[1].plot(self.data["ridgeback"]["odom"]["ts"], self.data["ridgeback"]["odom"]["omegas"][:, 0], label="wx")
        axes[1].plot(self.data["ridgeback"]["odom"]["ts"], self.data["ridgeback"]["odom"]["omegas"][:, 1], label="wy")
        axes[1].plot(self.data["ridgeback"]["odom"]["ts"], self.data["ridgeback"]["odom"]["omegas"][:, 2], label="wz")
        axes[1].set_title("Ridgeback Odom Angular Velocity")

        for ax in axes:
            ax.grid()
            ax.legend()

    def plot_base_velocity_estimation_odom(self):
        f, axes = plt.subplots(3,1,sharex=True)

        axes[0].plot(self.data["ridgeback"]["odom"]["ts"], self.data["ridgeback"]["odom"]["vs"][:, 0], 'r',label="v_x_odom")
        axes[1].plot(self.data["ridgeback"]["odom"]["ts"], self.data["ridgeback"]["odom"]["vs"][:, 1], 'r', label="v_y_odom")
        axes[2].plot(self.data["ridgeback"]["odom"]["ts"], self.data["ridgeback"]["odom"]["omegas"][:, 2], 'r', label="ω_odom")

        axes[0].plot(self.data["ridgeback"]["joint_states"]["ts"], self.data["ridgeback"]["joint_states"]["vbs"][:, 0], 'b', label="v_x_est")
        axes[1].plot(self.data["ridgeback"]["joint_states"]["ts"], self.data["ridgeback"]["joint_states"]["vbs"][:, 1], 'b', label="v_y_est")
        axes[2].plot(self.data["ridgeback"]["joint_states"]["ts"], self.data["ridgeback"]["joint_states"]["vbs"][:, 2], 'b', label="ω_est")

        axes[0].set_title("Ridgeback Velocity Estimation vs Odom (Base Frame)")


        for ax in axes:
            ax.grid()
            ax.legend()
        
        return axes

    def plot_base_velocity_estimation_vicon(self, axes=None, legend=""):
        f, axes = plt.subplots(3,1,sharex=True)

        axes[0].plot(self.data["vicon"]["ThingBase"]["ts"][:-1], self.data["vicon"]["ThingBase"]["pos_diff"][:, 0], 'r', label="v_x_vicon")
        axes[1].plot(self.data["vicon"]["ThingBase"]["ts"][:-1], self.data["vicon"]["ThingBase"]["pos_diff"][:, 1], 'r', label="v_y_vicon")
        axes[2].plot(self.data["vicon"]["ThingBase"]["ts"][:-1], self.data["vicon"]["ThingBase"]["yaw_diff"][:], 'r', label="ω_vicon")


        axes[0].plot(self.data["ridgeback"]["joint_states"]["ts"], self.data["ridgeback"]["joint_states"]["vs"][:, 0], 'b', label="v_x_est")
        axes[1].plot(self.data["ridgeback"]["joint_states"]["ts"], self.data["ridgeback"]["joint_states"]["vs"][:, 1], 'b', label="v_y_est")
        axes[2].plot(self.data["ridgeback"]["joint_states"]["ts"], self.data["ridgeback"]["joint_states"]["vs"][:, 2], 'b', label="ω_est")
        
        axes[0].set_title("Base Velocity Estimation vs Vicon")
        
        for i in range(3):
            axes[i].legend()
            axes[i].grid()

        return axes

    def plot_base_velocity_estimation_tf(self, axes=None, legend=""):
        f, axes = plt.subplots(3,1,sharex=True)

        axes[0].plot(self.data["tf"]["base_link"]["ts"][:-1], self.data["tf"]["base_link"]["pos_diff"][:, 0], 'r', label="v_x_tf")
        axes[1].plot(self.data["tf"]["base_link"]["ts"][:-1], self.data["tf"]["base_link"]["pos_diff"][:, 1], 'r', label="v_y_tf")
        axes[2].plot(self.data["tf"]["base_link"]["ts"][:-1], self.data["tf"]["base_link"]["rpy_diff"][:, 2], 'r', label="ω_tf")


        axes[0].plot(self.data["ridgeback"]["joint_states"]["ts"], self.data["ridgeback"]["joint_states"]["vs"][:, 0], 'b', label="v_x_est")
        axes[1].plot(self.data["ridgeback"]["joint_states"]["ts"], self.data["ridgeback"]["joint_states"]["vs"][:, 1], 'b', label="v_y_est")
        axes[2].plot(self.data["ridgeback"]["joint_states"]["ts"], self.data["ridgeback"]["joint_states"]["vs"][:, 2], 'b', label="ω_est")
        
        axes[0].set_title("Base Velocity Estimation vs TF")
        for i in range(3):
            axes[i].legend()
            axes[i].grid()
        
        return axes


    def plot_base_state_estimation_vicon(self, axes=None, legend=""):
        f, axes = plt.subplots(3,1,sharex=True)

        axes[0].plot(self.data["vicon"]["ThingBase"]["ts"], self.data["vicon"]["ThingBase"]["pos"][:, 0]-self.data["vicon"]["ThingBase"]["pos"][0, 0] + self.data["ridgeback"]["joint_states"]["qs"][0, 0], 'r', label="x_vicon")
        axes[1].plot(self.data["vicon"]["ThingBase"]["ts"], self.data["vicon"]["ThingBase"]["pos"][:, 1]-self.data["vicon"]["ThingBase"]["pos"][0, 1] + self.data["ridgeback"]["joint_states"]["qs"][0, 1], 'r', label="y_vicon")
        axes[2].plot(self.data["vicon"]["ThingBase"]["ts"], self.data["vicon"]["ThingBase"]["orn"]-self.data["vicon"]["ThingBase"]["orn"][0] + self.data["ridgeback"]["joint_states"]["qs"][0, 2], 'r', label="θ_vicon")
        
        axes[0].plot(self.data["ridgeback"]["joint_states"]["ts"], self.data["ridgeback"]["joint_states"]["qs"][:, 0], 'b', label="x_est")
        axes[1].plot(self.data["ridgeback"]["joint_states"]["ts"], self.data["ridgeback"]["joint_states"]["qs"][:, 1], 'b', label="y_est")
        axes[2].plot(self.data["ridgeback"]["joint_states"]["ts"], self.data["ridgeback"]["joint_states"]["qs"][:, 2], 'b', label="θ_est")
        
        for i in range(3):
            axes[i].legend()
            axes[i].grid()
        axes[0].set_title("Base Pose Estimation vs Vicon")

        return axes
    
    def plot_base_state_estimation_tf(self, axes=None, legend=""):
        f, axes = plt.subplots(3,1,sharex=True)

        axes[0].plot(self.data["tf"]["base_link"]["ts"], self.data["tf"]["base_link"]["pos"][:, 0], 'r', label="x_tf")
        axes[1].plot(self.data["tf"]["base_link"]["ts"], self.data["tf"]["base_link"]["pos"][:, 1], 'r', label="y_tf")
        axes[2].plot(self.data["tf"]["base_link"]["ts"], self.data["tf"]["base_link"]["rpy"][:, 2], 'r', label="θ_tf")
        
        axes[0].plot(self.data["ridgeback"]["joint_states"]["ts"], self.data["ridgeback"]["joint_states"]["qs"][:, 0], 'b', label="x_est")
        axes[1].plot(self.data["ridgeback"]["joint_states"]["ts"], self.data["ridgeback"]["joint_states"]["qs"][:, 1], 'b', label="y_est")
        axes[2].plot(self.data["ridgeback"]["joint_states"]["ts"], self.data["ridgeback"]["joint_states"]["qs"][:, 2], 'b', label="θ_est")
        
        
        for i in range(3):
            axes[i].legend()
            axes[i].grid()
        axes[0].set_title("Base Pose Estimation vs TF")

        return axes

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
                    self.data["ur10"]["joint_states"]["vs"][:, i], '-',linewidth=1.5,
                    label=r"$\dot\theta_{}$".format(i + 1))
            ax[i].plot(self.data["ur10"]["cmd_vels"]["ts"],
                    self.data["ur10"]["cmd_vels"]["vcs"][:, i], '-x',linewidth=1.5,
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
                   self.data["ridgeback"]["joint_states"]["vbs"][:, i], '-',linewidth=1.5,
                   label=labels_v[i])

            ax[i].plot(self.data["ridgeback"]["cmd_vels"]["ts"],
                   self.data["ridgeback"]["cmd_vels"]["vc_bs"][:, i], '-x',linewidth=1.5,
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
                    self.data["vicon"][VICON_TOOL_NAME]["pos"][:, i], '-',
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
    
    def plot_topic_frequency(self):
        self.plot_frequency(self.data["ridgeback"]["joint_states"], "joint states")
        self.plot_frequency(self.data["ridgeback"]["cmd_vels"], "cmd")
        self.plot_frequency(self.data["vicon"]["ThingBase"], "pos")
        # self.plot_frequency(self.data["tf"]["base_link"], "pos")

    def plot_frequency(self, data, name):
        f1 = plt.figure()
        dts = data["ts"][1:] - data["ts"][:-1]

        plt.plot(data["ts"][:-1], dts)
        plt.title(name + " time difference")
        
    def save_figs(self):

        figs = [plt.figure(n) for n in plt.get_fignums()]
        folder_name = str(self.data["dir_path"]).split("/")[-1]

        multipage(Path(str(self.data["folder_path"]))/Path(folder_name) / "report.pdf", figs)
    
    def plot_estimation(self):
        self.plot_base_state_estimation_tf()
        self.plot_base_state_estimation_vicon()
        
        self.plot_base_velocity_estimation_odom()
        self.plot_base_velocity_estimation_tf()
        self.plot_base_velocity_estimation_vicon()
        

class HTMPCDataPlotter():
    plotters: List[DataPlotter]

    def __init__(self, folder_path, data_plotter_class=DataPlotter):
        plotter_template = construct_logger(folder_path, process=False, data_plotter_class=data_plotter_class)
        task_num = plotter_template.data["mpc_solver_statuss"].shape[1]

        self.plotters = []
        mpc_ocp_params = [[] for i in range(task_num)]

        for task_id in range(task_num):
            p = construct_logger(folder_path, process=False, data_plotter_class=data_plotter_class)
            for key, value in p.data.items():
                print("Parsing {}".format(key))
                if "mpc" in key:
                    if "mpc_ocp_param" in key:
                        num = int(key.split("_")[-1][:-1])
                        if num == task_id:
                            mpc_ocp_params[task_id] = value
                    elif "sdf_param" in key:
                        pass
                    else:
                        p.data[key] = value[:, task_id]
                    print("HTMPC data. Filtering")
                else:
                    print("Shared data. Copied.")
            p.data["mpc_ocp_params"] = mpc_ocp_params[task_id]
            self.plotters.append(p)

        for p in self.plotters:
            p._post_processing()
            p._get_statistics()

        self.task_num = task_num
        self.controller = self.plotters[0].controller
        self.data = self.plotters[0].data

    
    def _get_multiplots(self):
        f, axes = plt.subplots(self.task_num, 1, sharex=True)

        return f, axes

    def plot_solver_status(self, axes=None, index=0, block=True, legend=None):
        if axes is None:
            _, axes = self._get_multiplots()
        
        for task_id, plotter in enumerate(self.plotters):
            plotter.plot_solver_status_htmpc([axes[task_id]], block=False or (block and task_id == self.task_num-1), legend=" ".join(["Task", str(task_id)]))
    
    def plot_cost(self, axes=None, index=0, block=True, legend=None):
        pass

    def plot_solver_iters(self, axes=None, index=0, block=True, legend=None):
        if axes is None:
            _, axes = self._get_multiplots()
        
        for task_id, plotter in enumerate(self.plotters):
            plotter.plot_solver_iters_htmpc([axes[task_id]], block=False or (block and task_id == self.task_num-1), legend=" ".join(["Task", str(task_id)]))

    def plot_time_controller(self, axes=None, index=0, block=True, legend=None):
        if axes is None:
            _, axes = self._get_multiplots()
        
        for task_id, plotter in enumerate(self.plotters):
            plotter.plot_time_htmpc([axes[task_id]], block=False or (block and task_id == self.task_num-1), legend=" ".join(["Task", str(task_id)]))

    def plot_mpc_prediction(self,data_name, task_id=1, t=0, block=True):
        self.plotters[task_id].plot_mpc_prediction(data_name, t, block=block, legend=" ".join(["Task", str(task_id)]), p_struct=self.controller.stmpc_p_structs[task_id])

    def plot_run_time(self, block=False):
        self.plotters[0].plot_run_time(block=block)

    def plot_mpc(self):


        self.plot_cost(block=False)
        self.plot_solver_status(block=False)

        self.plot_solver_iters(block=False)
        self.plot_time_controller(block=False)
        # naming convention: "mpc" + constraint_name + "constraint" + "predictins"

        self.plot_mpc_prediction("mpc_sdf_constraint_predictions",task_id=0,t=0.1, block=False)
        self.plot_mpc_prediction("mpc_sdf_constraint_predictions",task_id=1,t=0.1, block=False)

        self.plot_mpc_prediction("mpc_self_constraint_predictions", task_id=0, block=False)
        for task_id, stmpc_constraints in enumerate(self.controller.stmpc_constraints):
            for constraint in stmpc_constraints:
                if "Lex" in constraint.name:
                    self.plot_mpc_prediction("_".join(["mpc", constraint.name, "constraint", "predictions"]), task_id=task_id, block=False)

        # self.plot_mpc_prediction("mpc_control_constraint_predictions",block=False)
        # self.plot_mpc_prediction("mpc_state_constraint_predictions", block=False)
        self.plot_mpc_prediction("mpc_x_bars", task_id=0,t=0., block=False)
        self.plot_mpc_prediction("mpc_u_bars", task_id=0,t=0, block=False)

        self.plot_run_time()

        # mpc_failure_steps = np.where(self.data["mpc_solver_statuss"] == 4)[0]
        # if mpc_failure_steps.size > 0:
        #     self.mpc_constraint_debug(self.data["ts"][mpc_failure_steps[0]])
    
    def plot_robot(self):
        self.plotters[-1].plot_robot()
    
    def plot_tracking(self):
        self.plotters[-1].plot_tracking()
    
    def print_mpc_param(self, param_name):
        
        for task_id, plotter in enumerate(self.plotters):
            print("Task {}".format(task_id))
            plotter.print_mpc_param(param_name, self.controller.stmpc_p_structs[task_id])
    
    def savefigs(self):
        self.plotters[0].save_figs()
    
    def plot_sdf_map(self,t):
        self.plotters[0].plot_sdf_map(t=t, p_struct=self.controller.stmpc_p_structs[0])
    
    def plot_base_state_separate(self, axes=None, index=0, legend=None, linewidth=1, color=None):
        axes = self.plotters[-1].plot_base_state_separate(axes, index, legend,linewidth, color)
        return axes
    
    def plot_base_ref_separate(self, axes=None, index=0, legend=None, linewidth=1, color=None):
        axes = self.plotters[-1].plot_base_ref_separate(axes, index, legend,linewidth, color)
        return axes
    
    def plot_sdf_collision_separate(self, axes=None, index=0, color=None, linewidth=1, block=True, legend=None):
        axes = self.plotters[-1].plot_sdf_collision_separate(axes, index, color,linewidth, block, legend)
        return axes

    def plot_base_path(self, axes=None, index=0, legend=None, worldframe=True, linewidth=1, color=None):
        axes = self.plotters[-1].plot_base_path(axes, index, legend, worldframe, linewidth, color)
        return axes
    
    def plot_base_ref_path(self, axes=None, index=0, legend=None, worldframe=True, color='b'):
        axes = self.plotters[-1].plot_base_ref_path(axes, index, legend, worldframe, color)
        return axes

    def plot_exp(self, time):
        self.plot_time_controller(block=False)

        self.plot_mpc_prediction("mpc_x_bars", task_id=1,t=0., block=False)
        self.plot_mpc_prediction("mpc_u_bars", task_id=1,t=0, block=False)

        self.plot_run_time()
        self.plot_sdf_map(t=time)

        self.plot_tracking()



def construct_logger(path_to_folder, process=True, is_htmpc=False, data_plotter_class=DataPlotter, ht_data_plotter_class=HTMPCDataPlotter):
    """ Path to data folder

    :param path_to_folder:
    :return:
    """
    if is_htmpc:
        return data_plotter_class(path_to_folder, ht_data_plotter_class)
    else:
        items = os.listdir(path_to_folder)
        folder_num = 0
        file_num = 0
        for f in items:
            d = os.path.join(path_to_folder, f)
            if os.path.isdir(d):
                folder_num += 1
            else:
                file_num += 1

        if folder_num == 2:
            # if generated by running controller and simulator as two nodes
            # there will be two folders, one by controller and one by simulator
            return data_plotter_class.from_ROSSIM_results(path_to_folder, process)
        elif folder_num == 1 and file_num == 0:
            # if generated by running controller in loop with simulator,
            # there is only one folder
            return data_plotter_class.from_PYSIM_results(os.path.join(path_to_folder, items[0]), process)
        elif folder_num == 1 and file_num == 1:
            return data_plotter_class.from_ROSEXP_results(path_to_folder, process)