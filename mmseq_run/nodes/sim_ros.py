#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime
import logging
import time

import numpy as np
from pyb_utils.ghost import GhostSphere, GhostCylinder
import rospy

from mmseq_simulator import simulation
from mmseq_utils import parsing
from mmseq_utils.logging import DataLogger, DataPlotter
from mobile_manipulation_central.simulation_ros_interface import SimulatedMobileManipulatorROSInterface

def main():
    np.set_printoptions(precision=3, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration file.")
    parser.add_argument(
        "--video",
        nargs="?",
        default=None,
        const="",
        help="Record video. Optionally specify prefix for video directory.",
    )
    parser.add_argument("--GUI", action="store_true",
                        help="STMPC type, SQP or lex. This overwrites the yaml settings")
    args = parser.parse_args()

    # load configuration and overwrite with args
    config = parsing.load_config(args.config)

    if args.GUI:
        config["simulation"]["pybullet_connection"] = "GUI"

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
    logger = DataLogger(config)
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

        cmd_vel_world = robot.command_velocity(ros_interface.cmd_vel)
        t, _ = sim.step(t, step_robot=False)

        ee_curr_pos, _ = robot.link_pose()
        base_curr_pos, _ = robot.link_pose(-1)

        # log
        r_ew_w, Q_we = robot.link_pose()
        v_ew_w, ω_ew_w = robot.link_velocity()
        logger.append("ts", t)
        logger.append("xs", np.hstack((q,v)))
        logger.append("cmd_vels", cmd_vel_world)
        logger.append("r_ew_ws", r_ew_w)
        logger.append("Q_wes", Q_we)
        logger.append("v_ew_ws", v_ew_w)
        logger.append("ω_ew_ws", ω_ew_w)

        logger.append("r_bw_ws", q[:2])


        time.sleep(sim.timestep)

    timestamp = datetime.datetime.now()
    logger.save(timestamp, "data")
    # plotter = DataPlotter.from_logger(logger)
    # plotter.plot_cmd_vs_real_vel()
    # plotter.plot_ee_position()
    # plotter.show()

if __name__ == "__main__":
    main()
