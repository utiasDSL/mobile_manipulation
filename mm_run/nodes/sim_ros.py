import argparse
import datetime
import os
import sys
import time

import numpy as np
import rospy
from mobile_manipulation_central.simulation_ros_interface import (
    SimulatedMobileManipulatorROSInterface,
    SimulatedViconObjectInterface,
)

from mm_simulator import simulation
from mm_utils import parsing
from mm_utils.logging import DataLogger


def main():
    np.set_printoptions(precision=3, suppress=True)
    argv = rospy.myargv(argv=sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration file.")
    parser.add_argument(
        "--video",
        nargs="?",
        default=None,
        const="",
        help="Record video. Optionally specify prefix for video directory.",
    )
    parser.add_argument(
        "--logging_sub_folder",
        type=str,
        help="save data in a sub folder of logging directory",
    )
    parser.add_argument(
        "--GUI",
        action="store_true",
        help="Enable PyBullet GUI. This overwrites the yaml settings",
    )
    args = parser.parse_args(argv[1:])

    # load configuration and overwrite with args
    config = parsing.load_config(args.config)

    if args.GUI:
        config["simulation"]["gui"] = True

    if args.logging_sub_folder != "default":
        config["logging"]["log_dir"] = os.path.join(
            config["logging"]["log_dir"], args.logging_sub_folder
        )

    sim_config = config["simulation"]

    # start the simulation
    timestamp = datetime.datetime.now()
    sim = simulation.BulletSimulation(
        config=sim_config, timestamp=timestamp, cli_args=args
    )
    robot = sim.robot

    # initial time, state, input
    t = 0.0

    # init logger
    logger = DataLogger(config, name="sim")
    logger.add("sim_timestep", sim.timestep)
    logger.add("duration", sim.duration)

    logger.add("nq", sim_config["robot"]["dims"]["q"])
    logger.add("nv", sim_config["robot"]["dims"]["v"])
    logger.add("nx", sim_config["robot"]["dims"]["x"])
    logger.add("nu", sim_config["robot"]["dims"]["u"])

    # ros interface
    rospy.init_node("sim_ros")
    ros_interface = SimulatedMobileManipulatorROSInterface()
    ros_interface.publish_time(t)

    vicon_tool_interface = SimulatedViconObjectInterface(
        sim_config["robot"]["tool_vicon_name"]
    )
    while not ros_interface.ready():
        q, v = robot.joint_states()
        ros_interface.publish_feedback(t, q, v)
        ros_interface.publish_time(t)
        t += sim.timestep
        time.sleep(sim.timestep)

        if rospy.is_shutdown():
            return

    print("Control commands received. Proceed ... ")
    t0 = t
    while not rospy.is_shutdown() and t - t0 <= sim.duration:
        q, v = robot.joint_states()
        ros_interface.publish_feedback(t, q, v)
        ros_interface.publish_time(t)

        cmd_vel_world = robot.command_velocity(ros_interface.cmd_vel, bodyframe=True)
        ee_curr_pos, ee_curr_orn = robot.link_pose()
        base_curr_pos, _ = robot.link_pose(-1)
        vicon_tool_interface.publish_pose(t, ee_curr_pos, ee_curr_orn)

        # log
        r_ew_w, Q_we = robot.link_pose()
        v_ew_w, ω_ew_w = robot.link_velocity()
        logger.append("ts", t)
        logger.append("xs", np.hstack((q, v)))
        logger.append("cmd_vels", cmd_vel_world)
        logger.append("r_ew_ws", r_ew_w)
        logger.append("Q_wes", Q_we)
        logger.append("v_ew_ws", v_ew_w)
        logger.append("ω_ew_ws", ω_ew_w)

        logger.append("r_bw_ws", q[:2])
        logger.append("yaw_bw_ws", q[2])
        logger.append("v_bw_ws", v[:2])
        logger.append("ω_bw_ws", v[2])

        t, _ = sim.step(t)
        time.sleep(sim.timestep)

    logger.save()


if __name__ == "__main__":
    main()
