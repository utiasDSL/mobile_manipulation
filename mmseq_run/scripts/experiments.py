#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime
import logging
import time

import numpy as np
from spatialmath.base import rotz

import mmseq_control.HTMPC as HTMPC
import mmseq_control.HTIDKC as HTIDKC
import mmseq_plan.TaskManager as TaskManager
from mmseq_simulator import simulation
from mmseq_utils import parsing
from mmseq_utils.logging import DataLogger, DataPlotter


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
    parser.add_argument("--type", type=str, default=None,
                        help="controller type, HTMPCSQP or HTMPCLex. This overwrites the yaml settings")
    parser.add_argument("--GUI", action="store_true",
                        help="Pybullet GUI. This overwrites the yaml settings")
    args = parser.parse_args()

    # load configuration and overwrite with args
    config = parsing.load_config(args.config)
    if args.type is not None:
        config["controller"]["type"] = args.stmpctype

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
        control_class = getattr(HTIDKC, ctrl_config["type"], None)
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
        if "MPC" in ctrl_config["type"]:
            _, acc, mpc_plan = results
            u += mpc_plan[0] * sim.timestep
        else:
            u, acc = results
        t1 = time.perf_counter()

        robot.command_velocity(u)
        t, _ = sim.step(t, step_robot=False)
        ee_curr_pos, ee_cur_orn = robot.link_pose()
        base_curr_pos, _ = robot.link_pose(-1)
        states = {"base": robot_states[0][:3],
                  "EE": (ee_curr_pos, ee_cur_orn)}
        sot.update(t, states)

        # log
        r_ew_w, Q_we = robot.link_pose()
        v_ew_w, ω_ew_w = robot.link_velocity()
        r_ew_wd = []
        r_bw_wd = []
        for planner in planners:
            if planner.type == "EE":
                r_ew_wd, _ = planner.getTrackingPoint(t, robot_states)
                print(r_ew_wd)
            elif planner.type == "base":
                r_bw_wd, _ = planner.getTrackingPoint(t, robot_states)
        logger.append("ts", t)
        logger.append("xs", np.hstack(robot_states))
        logger.append("controller_run_time", t1 - t0)
        logger.append("cmd_vels", u)
        logger.append("cmd_accs", acc)
        if len(r_ew_wd)>0:
            logger.append("r_ew_w_ds", r_ew_wd)
        logger.append("r_ew_ws", r_ew_w)
        logger.append("Q_wes", Q_we)
        logger.append("v_ew_ws", v_ew_w)
        logger.append("ω_ew_ws", ω_ew_w)

        if len(r_bw_wd)>0:
            logger.append("r_bw_w_ds", r_bw_wd)
        logger.append("r_bw_ws", robot_states[0][:2])

        # if controller.solver_status is not None:
        #     logger.append("mpc_solver_statuss", controller.solver_status)
        # logger.append("mpc_cost_iters", controller.cost_iter)
        # logger.append("mpc_cost_finals", controller.cost_final)
        # if controller.step_size is not None:
        #     logger.append("mpc_step_sizes", controller.step_size)

        # if controller.stmpc_run_time is not None:
        #     logger.append("stmpc_run_time", controller.stmpc_run_time)

        print("Time {}".format(t))
        time.sleep(sim.timestep)

    data_name = ctrl_config["type"]
    timestamp = datetime.datetime.now()
    logger.save(timestamp, data_name)

if __name__ == "__main__":
    main()
