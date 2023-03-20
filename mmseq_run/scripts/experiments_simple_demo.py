#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import time
import datetime

import numpy as np
import matplotlib.pyplot as plt
from pyb_utils.ghost import GhostSphere, GhostCylinder

from mmseq_control.HTMPC import HTMPC, HTMPCLex
from mmseq_control.IDKC import IKCPrioritized
from mmseq_simulator import simulation
from mmseq_plan.TaskManager import SoTCycle
from mmseq_utils import parsing
from mmseq_utils.logging import DataLogger, DataPlotter

class EventHandler:
    def __init__(self, sot):
        self.sot = sot

    def onclick(self, event):
        self.sot.shuffle_is_triggered = True




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
    parser.add_argument("--priority", type=str, default=None, help="priority, EE or base")
    parser.add_argument("--stmpctype", type=str, default=None,
                        help="STMPC type, SQP or lex. This overwrites the yaml settings")
    parser.add_argument("--GUI", action="store_true",
                        help="STMPC type, SQP or lex. This overwrites the yaml settings")
    args = parser.parse_args()

    # load configuration and overwrite with args
    config = parsing.load_config(args.config)
    if args.stmpctype is not None:
        config["controller"]["type"] = args.stmpctype
    if args.priority is not None:
        config["planner"]["priority"] = args.priority

    if args.GUI:
        config["simulation"]["pybullet_connection"] = "GUI"

    if config["controller"]["type"] == "lex":
        config["controller"]["HT_MaxIntvl"] = 1

    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    planner_config = config["planner"]

    # start the simulation
    timestamp = datetime.datetime.now()
    sim = simulation.BulletSimulation(
        config=sim_config, timestamp=timestamp, cli_args=args
    )
    robot = sim.robot
    # robot.command_velocity(np.zeros(9))
    # sim.settle(5.0)

    if ctrl_config["type"] == "SQP" or ctrl_config["type"] == "SQP_TOL_SCHEDULE":
        controller = HTMPC(ctrl_config)
    elif ctrl_config["type"] == "lex":
        controller = HTMPCLex(ctrl_config)
    elif ctrl_config["type"] == "TP-IDKC":
        controller = IKCPrioritized(ctrl_config)
    sot = SoTCycle(planner_config)

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
    colors = [[1, 0, 0, 1],
              [0, 1, 0, 1],
              [0, 0, 1, 1]]
    for pid, planner in enumerate(sot.planners):
        pd = planner.target_pos
        print(planner.name)
        if planner.type == "EE":
            sim.ghosts.append(GhostSphere(planner.tracking_err_tol, pd, color=colors[pid//2]))
        else:
            sim.ghosts.append(GhostCylinder(position=np.hstack((pd, 0)), color=colors[pid//2]))

    # interaction pad
    fig = plt.figure(figsize=[10, 10], dpi=300)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    handler = EventHandler(sot)
    cid = fig.canvas.mpl_connect('button_press_event', handler.onclick)
    plt.show(block=False)
    plt.pause(2)
    while t <= sim.duration:
        # open-loop command
        robot_states = robot.joint_states(add_noise=False)

        # print(robot_states[0])
        t0 = time.perf_counter()

        planners = sot.getPlanners(num_planners=2)
        u, acc = controller.control(t, robot_states, planners)
        t1 = time.perf_counter()
        print(t1-t0)
        robot.command_velocity(u)
        t, _ = sim.step(t, step_robot=False)
        plt.show(block=False)
        plt.pause(0.01)

        states = {"EE": robot.link_pose(), "base": robot_states[0][:3]}
        sot.update(t, states)

        # update ghost object in sim
        for pid in range(sot.planner_num):
            p_indx = (sot.curr_task_id + pid - sot.curr_task_id % 2) % sot.planner_num

            pd = sot.planners[p_indx].target_pos
            if pd.size == 2:
                pd = np.hstack((pd, 0))
            sim.ghosts[pid].update(position=pd)


        ee_curr_pos, _ = robot.link_pose()
        base_curr_pos, _ = robot.link_pose(-1)

        # log
        r_ew_w, Q_we = robot.link_pose()
        v_ew_w, ω_ew_w = robot.link_velocity()
        for planner in planners:
            if planner.type == "EE":
                r_ew_wd, _ = planner.getTrackingPoint(t, robot_states)
            elif planner.type == "base":
                r_bw_wd, _ = planner.getTrackingPoint(t, robot_states)
        logger.append("ts", t)
        logger.append("xs", np.hstack(robot_states))
        logger.append("cmd_vels", u)
        logger.append("r_ew_w_ds", r_ew_wd)
        logger.append("r_ew_ws", r_ew_w)
        logger.append("Q_wes", Q_we)
        logger.append("v_ew_ws", v_ew_w)
        logger.append("ω_ew_ws", ω_ew_w)

        logger.append("r_bw_w_ds", r_bw_wd)
        logger.append("r_bw_ws", robot_states[0][:2])

        if ctrl_config["type"] == "SQP" or ctrl_config["type"] == "SQP_TOL_SCHEDULE":
            logger.append("mpc_solver_statuss", controller.solver_status)
            logger.append("mpc_cost_iters", controller.cost_iter)
            logger.append("mpc_cost_finals", controller.cost_final)
            logger.append("mpc_step_sizes", controller.step_size)

        time.sleep(sim.timestep)

    data_name = ctrl_config["type"]
    timestamp = datetime.datetime.now()
    logger.save(timestamp, data_name)
    # plotter = DataPlotter.from_logger(logger)
    # plotter.plot_cmd_vs_real_vel()
    # plotter.plot_ee_position()
    # plotter.show()

if __name__ == "__main__":
    main()
