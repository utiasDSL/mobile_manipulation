#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime
import logging
import time
import os

import numpy as np
from spatialmath.base import rotz


import mmseq_plan.TaskManager as TaskManager
from mmseq_control.mobile_manipulator_point_mass.mobile_manipulator_class import MobileManipulatorPointMass
from mmseq_control.robot import CasadiModelInterface
import mmseq_plan.CPCPlanner as CPCPlanner
import mmseq_plan.SequentialPlanner as SequentialPlanner
from mmseq_simulator import simulation
from mmseq_utils import parsing
from mmseq_utils.logging import DataLogger
from pyb_utils import debug_frame_world


def planner_coord_transform(q, ree, planners):
    R_wb = rotz(q[2])
    for planner in planners:
        P = np.zeros(3)
        if planner.frame_id == "base":
            P = np.hstack((q[:2], 0))
        elif planner.frame_id == "EE":
            P = ree

        if planner.__class__.__name__ == "EESimplePlanner":
            planner.target_pos = R_wb @ planner.target_pos + P
        elif planner.__class__.__name__ == "EEPosTrajectoryCircle":
            planner.c = R_wb @ planner.c + P
            planner.plan["p"] = planner.plan["p"] @ R_wb.T + P

        elif planner.__class__.__name__ == "BaseSingleWaypoint":
            planner.target_pos = (R_wb @ np.hstack((planner.target_pos, 0)))[:2] + P[:2]
        elif planner.__class__.__name__ == "BasePosTrajectoryCircle":
            planner.c = R_wb[:2, :2] @ planner.c + P[:2]
            planner.plan["p"] = planner.plan["p"] @ R_wb[:2, :2].T + P[:2]
        elif planner.__class__.__name__ == "BasePosTrajectoryLine":
            planner.plan["p"] = planner.plan["p"] @ R_wb[:2, :2].T + P[:2]

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
                        help="Pybullet GUI. This overwrites the yaml settings")
    args = parser.parse_args() 

    # load configuration and overwrite with args
    config = parsing.load_config(args.config)

    if args.GUI:
        config["simulation"]["pybullet_connection"] = "GUI"


    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    planner_config = config["planner"]

    # Simulator
    timestamp = datetime.datetime.now()
    sim = simulation.BulletSimulation(
        config=sim_config, timestamp=timestamp, cli_args=args
    )
    robot = sim.robot

    # Try no controller, directly input velocities in simulation
    # here probably put the code for generating trajectory
    # initialize robot (needed for generating trajectory)

    # initialize robot
    planner_config = config["planner"]["tasks"][0]

    planner = getattr(CPCPlanner, planner_config["planner_type"], None)
    optimization_type = 'cpc'
    if planner is None:
        planner = getattr(SequentialPlanner, planner_config["planner_type"], None)
        optimization_type = 'sequential'
    if planner is None:
        raise ValueError("Planner type {} not found".format(planner_config["planner_type"]))
    # generate plan
    planner = planner(planner_config)
    planner.generatePlanFromConfig()

    # set py logger level
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    sim_log = logging.getLogger("Simulator")
    sim_log.setLevel(config["logging"]["log_level"])
    sim_log.addHandler(ch)

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

    log = True
    finished = False
    u = np.zeros(sim_config["robot"]["dims"]["v"])

    while t <= sim.duration:
        # open-loop command
        robot_states = robot.joint_states(add_noise=False)

        t0 = time.perf_counter()
        u = planner.control(t)
        # print(u)
        t1 = time.perf_counter()

        robot.command_velocity(u)
        t, _ = sim.step(t, step_robot=False)
        ee_curr_pos, ee_cur_orn = robot.link_pose()
        base_curr_pos, _ = robot.link_pose(-1)
        ee_states = (ee_curr_pos, ee_cur_orn)
        states = {"base": (robot_states[0][:3], robot_states[1][:3]), "EE": ee_states}


        # log
        r_ew_w, Q_we = robot.link_pose()
        v_ew_w, ω_ew_w = robot.link_velocity()
        r_ew_wd = []
        r_bw_wd = []
        
        logger.append("ts", t)
        logger.append("xs", np.hstack(robot_states))
        logger.append("controller_run_time", t1 - t0)
        logger.append("cmd_vels", u)
        # logger.append("cmd_accs", acc)

        logger.append("r_ew_ws", r_ew_w)
        logger.append("Q_wes", Q_we)
        logger.append("v_ew_ws", v_ew_w)
        logger.append("ω_ew_ws", ω_ew_w)
        logger.append("r_bw_ws", robot_states[0][:2])

        if len(r_ew_wd)>0:
            logger.append("r_ew_w_ds", r_ew_wd)
        if len(r_bw_wd)>0:
            logger.append("r_bw_w_ds", r_bw_wd)

        sim_log.log(20, "Time {}".format(t))
        time.sleep(sim.timestep)

    data_name = ctrl_config["type"]
    timestamp = datetime.datetime.now()
    logger.save(timestamp, data_name)

if __name__ == "__main__":
    main()
