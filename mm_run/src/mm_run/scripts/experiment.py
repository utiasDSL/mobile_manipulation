import argparse
import datetime
import logging
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import mm_control.MPC as MPC
from mm_plan.TaskManager import TaskManager
from mm_simulator import simulation
from mm_utils import parsing
from mm_utils.logging import DataLogger
from mm_utils.math import compute_velocity_command
from mm_utils.metrics import extract_robot_states


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
    planner_config = config.get("planner", None)

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

    # Controller frequency management
    ctrl_period = 1.0 / ctrl_config.get("ctrl_rate")
    last_controller_time = -ctrl_period  # Initialize to allow first call

    t = 0.0
    while t <= sim.duration:
        print(f"-------------- {t:.3f}s/{sim.duration}s ------------------")
        # open-loop command
        robot_states = robot.joint_states(add_noise=False)

        # Only call controller if enough time has passed
        if t - last_controller_time >= ctrl_period:
            # Get references from TaskManager
            references = sot.getReferences(
                t, robot_states, controller.N + 1, controller.dt
            )

            t0 = time.perf_counter()
            v_bar, u_bar = controller.control(t, robot_states, references)
            t1 = time.perf_counter()
            controller_log.log(20, f"Controller Run Time: {t1 - t0}")
            last_controller_time = t

        u = compute_velocity_command(
            u,
            u_bar,
            v_bar,
            ctrl_config["cmd_vel_type"],
            t - last_controller_time,
            controller.dt,
            simulation_dt=sim.timestep,
        )

        robot.command_velocity(u)
        t, _ = sim.step(t)

        # Extract robot states using shared utility function
        states = extract_robot_states(robot, robot_states)

        # Pass MPC masks to TaskManager for task completion checks
        sot.update(
            t, states, base_mask=controller.base_mask, ee_mask=controller.ee_mask
        )

        # log
        # Use extracted states instead of calling link_velocity() again
        v_ew_w = states["EE"]["velocity"][:3]
        ω_ew_w = states["EE"]["velocity"][3:]

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
        logger.append("r_ew_ws", states["EE"]["pose"][:3])
        # Convert Euler angles back to quaternion for logging
        ee_quat = Rot.from_euler("xyz", states["EE"]["pose"][3:]).as_quat()
        logger.append("Q_wes", ee_quat)
        logger.append("v_ew_ws", v_ew_w)
        logger.append("ω_ew_ws", ω_ew_w)
        logger.append("r_bw_ws", states["base"]["pose"][:2])

        if r_bw_wd is not None:
            if r_bw_wd.shape[0] == 2:
                logger.append("r_bw_w_ds", r_bw_wd)
            elif r_bw_wd.shape[0] == 3:
                logger.append("r_bw_w_ds", r_bw_wd[:2])
                logger.append("yaw_bw_w_ds", r_bw_wd[2])
                logger.append("yaw_bw_ws", states["base"]["pose"][2])
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

    session_timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    logger.save(session_timestamp=session_timestamp)


if __name__ == "__main__":
    main()
