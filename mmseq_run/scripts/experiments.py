#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime
import logging
import time
import os

import numpy as np
from spatialmath.base import rotz

import mmseq_control.HTMPC as HTMPC
import mmseq_control.HTIDKC as HTIDKC
import mmseq_control.STMPC as STMPC
import mmseq_control_new.MPC as MPC
import mmseq_control_new.HTMPC as HTMPCNew
import mmseq_plan.TaskManager as TaskManager
from mmseq_simulator import simulation
from mmseq_utils import parsing
from mmseq_utils.logging import DataLogger


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
    parser.add_argument("--ctrl_config", type=str, default="default",
                        help="controller config. This overwrites the yaml settings in config if not set to default")
    parser.add_argument("--planner_config", type=str, default="default",
                        help="plannner config. This overwrites the yaml settings in config if not set to default")
    parser.add_argument("--logging_sub_folder", type=str, default="default",
                        help="save data in a sub folder of logging director")
    parser.add_argument("--GUI", action="store_true",
                        help="Pybullet GUI. This overwrites the yaml settings")
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
        config["logging"]["log_dir"] = os.path.join(config["logging"]["log_dir"], args.logging_sub_folder)

    if args.GUI:
        config["simulation"]["pybullet_connection"] = "GUI"

    if config["controller"]["type"] == "HTMPCLex":
        config["controller"]["HT_MaxIntvl"] = 1

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
    control_class = getattr(HTMPC, ctrl_config["type"], None)
    if control_class is None:
        control_class = getattr(STMPC, ctrl_config["type"], None)
    if control_class is None:
        control_class = getattr(MPC, ctrl_config["type"], None)
    if control_class is None:
        control_class = getattr(HTMPCNew, ctrl_config["type"], None)

    controller = control_class(ctrl_config)

    # Stack of Tasks
    sot_class = getattr(TaskManager, planner_config["sot_type"])
    sot = sot_class(planner_config)
    planner_coord_transform(robot.joint_states()[0], robot.link_pose()[0], sot.planners)

    # set py logger level
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    sot.activatePlanners()
    u = np.zeros(sim_config["robot"]["dims"]["v"])

    while t <= sim.duration:
        # open-loop command
        robot_states = robot.joint_states(add_noise=False)
        planners = sot.getPlanners(num_planners=2)

        t0 = time.perf_counter()
        results = controller.control(t, robot_states, planners)
        t1 = time.perf_counter()
        controller_log.log(20, "Controller Run Time: {}".format(t1 - t0))

        if "MPC" in ctrl_config["type"]:
            _, acc, u_bar, v_bar = results
            u += u_bar[0] * sim.timestep
        else:
            u, acc = results

        robot.command_velocity(u)
        t, _ = sim.step(t, step_robot=False)
        ee_curr_pos, ee_cur_orn = robot.link_pose()
        base_curr_pos, _ = robot.link_pose(-1)
        ee_states = (ee_curr_pos, ee_cur_orn)
        states = {"base": (robot_states[0][:3], robot_states[1][:3]), "EE": ee_states}

        if sot.__class__.__name__ == "SoTSequentialTasks":
            if t > 12 and t < 12.2:
                human_pos = np.array([[-3, -3, 0.8],
                                      [-3, 3, 0.8],
                                      [3, -0, 1.0]])
                # self.sot.update_planner(self.vicon_marker_swarm_interface.position, states)
                sot.update_planner(human_pos, states)

        sot.update(t, states)

        # log
        r_ew_w, Q_we = robot.link_pose()
        v_ew_w, ω_ew_w = robot.link_velocity()
        r_ew_wd = []
        r_bw_wd = []
        for planner in planners:
            if planner.type == "EE":
                r_ew_wd, _ = planner.getTrackingPoint(t, robot_states)
            elif planner.type == "base":
                r_bw_wd, _ = planner.getTrackingPoint(t, robot_states)
        logger.append("ts", t)
        logger.append("xs", np.hstack(robot_states))
        logger.append("controller_run_time", t1 - t0)
        logger.append("cmd_vels", u)
        logger.append("cmd_accs", acc)

        logger.append("r_ew_ws", r_ew_w)
        logger.append("Q_wes", Q_we)
        logger.append("v_ew_ws", v_ew_w)
        logger.append("ω_ew_ws", ω_ew_w)
        logger.append("r_bw_ws", robot_states[0][:2])

        if len(r_ew_wd)>0:
            logger.append("r_ew_w_ds", r_ew_wd)
        if len(r_bw_wd)>0:
            logger.append("r_bw_w_ds", r_bw_wd)
        if "MPC" in ctrl_config["type"]:
            for (key, val) in controller.log.items():
                    logger.append("_".join(["mpc", key])+"s", val)

        sim_log.log(20, "Time {}".format(t))
        time.sleep(sim.timestep)

    data_name = ctrl_config["type"]
    timestamp = datetime.datetime.now()
    logger.save(timestamp, data_name)

if __name__ == "__main__":
    main()
