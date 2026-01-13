import argparse
import datetime
import logging
import os
import time

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as Rot

import mm_control.MPC as MPC
from mm_plan.TaskManager import TaskManager
from mm_simulator import simulation
from mm_utils import parsing
from mm_utils.logging import DataLogger


def main():
    np.set_printoptions(precision=3, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, help="Path to configuration file."
    )
    parser.add_argument(
        "--video",
        nargs="?",
        default=None,
        const="",
        help="Record video. Optionally specify prefix for video directory.",
    )
    parser.add_argument(
        "--ctrl_config",
        type=str,
        default="default",
        help="controller config. This overwrites the yaml settings in config if not set to default",
    )
    parser.add_argument(
        "--planner_config",
        type=str,
        default="default",
        help="planner config. This overwrites the yaml settings in config if not set to default",
    )
    parser.add_argument(
        "--logging_sub_folder",
        type=str,
        default="default",
        help="save data in a sub folder of logging directory",
    )
    parser.add_argument(
        "--GUI",
        action="store_true",
        help="Pybullet GUI. This overwrites the yaml settings",
    )
    args = parser.parse_args()

    # load configuration and overwrite with args
    config = parsing.load_config(args.config)
    if args.ctrl_config != "default":
        ctrl_config = parsing.load_config(args.ctrl_config)
        config = parsing.recursive_dict_update(config, ctrl_config)
    if args.planner_config != "default":
        planner_config = parsing.load_config(args.planner_config)
        config = parsing.recursive_dict_update(config, planner_config)

    if args.logging_sub_folder != "default":
        config["logging"]["log_dir"] = os.path.join(
            config["logging"]["log_dir"], args.logging_sub_folder
        )

    if args.GUI:
        config["simulation"]["gui"] = True

    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    planner_config = config["planner"]

    # Simulator
    timestamp = datetime.datetime.now()
    sim = simulation.BulletSimulation(
        config=sim_config, timestamp=timestamp, cli_args=args
    )
    robot = sim.robot

    # Controller
    control_class = getattr(MPC, ctrl_config["type"], None)
    if control_class is None:
        raise ValueError(f"Unknown controller type: {ctrl_config['type']}")

    controller = control_class(ctrl_config)

    # Task Manager (simplified - only sequential execution)
    sot = TaskManager(planner_config)

    # set py logger level
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    planner_log = logging.getLogger("Planner")
    planner_log.setLevel(config["logging"]["log_level"])
    planner_log.addHandler(ch)
    controller_log = logging.getLogger("Controller")
    controller_log.setLevel(config["logging"]["log_level"])
    controller_log.addHandler(ch)
    sim_log = logging.getLogger("Simulator")
    sim_log.setLevel(config["logging"]["log_level"])
    sim_log.addHandler(ch)

    # initial time, state, input
    t = 0.0

    # init logger (combined sim+control in one process)
    logger = DataLogger(config, name="combined")

    logger.add("sim_timestep", sim.timestep)
    logger.add("duration", sim.duration)

    logger.add("nq", sim_config["robot"]["dims"]["q"])
    logger.add("nv", sim_config["robot"]["dims"]["v"])
    logger.add("nx", sim_config["robot"]["dims"]["x"])
    logger.add("nu", sim_config["robot"]["dims"]["u"])

    sot.activatePlanners()
    u = np.zeros(sim_config["robot"]["dims"]["v"])

    while t <= sim.duration:
        # open-loop command
        robot_states = robot.joint_states(add_noise=False)

        # Get references from TaskManager
        references = sot.getReferences(t, robot_states, controller.N + 1, controller.dt)

        t0 = time.perf_counter()
        v_bar, u_bar = controller.control(t, robot_states, references)
        t1 = time.perf_counter()
        controller_log.log(20, f"Controller Run Time: {t1 - t0}")

        if ctrl_config["cmd_vel_type"] == "integration":
            u += u_bar[0] * sim.timestep
        elif ctrl_config["cmd_vel_type"] == "interpolation":
            # Interpolate velocity trajectory at sim.timestep
            # v_bar has shape (N+1, nu), times are at 0, dt, 2*dt, ..., N*dt
            N = v_bar.shape[0]
            t_v_bar = np.arange(N) * controller.dt
            v_interp = interp1d(
                t_v_bar,
                v_bar,
                axis=0,
                bounds_error=False,
                fill_value="extrapolate",
            )
            u = v_interp(sim.timestep)
        else:
            raise ValueError(f"Unknown cmd_vel_type: {ctrl_config['cmd_vel_type']}")

        robot.command_velocity(u)
        t, _ = sim.step(t)

        # Convert to pose arrays in world frame
        ee_curr_pos, ee_cur_orn = robot.link_pose()
        ee_euler = Rot.from_quat(ee_cur_orn).as_euler("xyz")
        ee_pose = np.hstack([ee_curr_pos, ee_euler])

        ee_lin_vel, ee_ang_vel = robot.link_velocity()
        ee_vel = np.hstack([ee_lin_vel, ee_ang_vel])  # [vx, vy, vz, wx, wy, wz]

        base_pose = robot_states[0][:3]  # [x, y, yaw] already in world frame
        base_vel = robot_states[1][:3]  # [vx, vy, vyaw]

        states = {
            "base": {"pose": base_pose, "velocity": base_vel},
            "EE": {"pose": ee_pose, "velocity": ee_vel},
        }

        sot.update(t, states)

        # log
        v_ew_w, ω_ew_w = robot.link_velocity()

        # Get tracking points from references
        r_ew_wd = None
        r_bw_wd = None
        v_ew_wd = None
        v_bw_wd = None

        if references.get("ee_pose") is not None:
            r_ew_wd = references["ee_pose"][0][:3]  # Current EE position reference
            if references.get("ee_velocity") is not None:
                # Current EE linear velocity reference
                v_ew_wd = references["ee_velocity"][0][:3]

        if references.get("base_pose") is not None:
            r_bw_wd = references["base_pose"][0]  # Current base pose reference
            if references.get("base_velocity") is not None:
                # Current base velocity reference
                v_bw_wd = references["base_velocity"][0]

        logger.append("ts", t)
        logger.append("xs", np.hstack(robot_states))
        logger.append("controller_run_time", t1 - t0)
        logger.append("cmd_vels", u)
        logger.append("r_ew_ws", ee_curr_pos)
        logger.append("Q_wes", ee_cur_orn)
        logger.append("v_ew_ws", v_ew_w)
        logger.append("ω_ew_ws", ω_ew_w)
        logger.append("r_bw_ws", robot_states[0][:2])

        if r_bw_wd is not None:
            if r_bw_wd.shape[0] == 2:
                logger.append("r_bw_w_ds", r_bw_wd)
            elif r_bw_wd.shape[0] == 3:
                logger.append("r_bw_w_ds", r_bw_wd[:2])
                logger.append("yaw_bw_w_ds", r_bw_wd[2])
                logger.append("yaw_bw_ws", robot_states[0][2])
        if v_bw_wd is not None:
            if v_bw_wd.shape[0] == 2:
                logger.append("v_bw_w_ds", v_bw_wd)
            elif v_bw_wd.shape[0] == 3:
                logger.append("v_bw_w_ds", v_bw_wd[:2])
                logger.append("ω_bw_w_ds", v_bw_wd[2])
        if r_ew_wd is not None:
            logger.append("r_ew_w_ds", r_ew_wd)
        if v_ew_wd is not None:
            logger.append("v_ew_w_ds", v_ew_wd)
        if "MPC" in ctrl_config["type"]:
            for key, val in controller.log.items():
                logger.append("_".join(["mpc", key]) + "s", val)

        time.sleep(sim.timestep)

    session_timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    logger.save(session_timestamp=session_timestamp)


if __name__ == "__main__":
    main()
