#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime
import logging
import time
import sys

import numpy as np
import rospy
from spatialmath.base import rotz

from mmseq_control.HTMPC import HTMPC, HTMPCLex
from mmseq_simulator import simulation
from mmseq_plan.TaskManager import SoTStatic
from mmseq_utils import parsing
from mmseq_utils.logging import DataLogger, DataPlotter
from mobile_manipulation_central.ros_interface import MobileManipulatorROSInterface, ViconObjectInterface

class ControllerROSNode:

    def __init__(self):

        np.set_printoptions(precision=3, suppress=True)
        argv = rospy.myargv(argv=sys.argv)
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True, help="Path to configuration file.")
        parser.add_argument("--priority", type=str, default=None, help="priority, EE or base")
        parser.add_argument("--stmpctype", type=str, default=None,
                            help="STMPC type, SQP or lex. This overwrites the yaml settings")
        args = parser.parse_args(argv[1:])


        # load configuration and overwrite with args
        config = parsing.load_config(args.config)
        if args.stmpctype is not None:
            config["controller"]["type"] = args.stmpctype
        if args.priority is not None:
            config["planner"]["priority"] = args.priority

        if config["controller"]["type"] == "lex":
            config["controller"]["HT_MaxIntvl"] = 1

        ctrl_config = config["controller"]
        self.planner_config = config["planner"]

        # controller
        if ctrl_config["type"] == "SQP" or ctrl_config["type"] == "SQP_TOL_SCHEDULE":
            self.controller = HTMPC(ctrl_config)
        elif ctrl_config["type"] == "lex":
            self.controller = HTMPCLex(ctrl_config)

        self.ctrl_rate = ctrl_config["rate"]

        # set py logger level
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.planner_log = logging.getLogger("Planner")
        self.planner_log.setLevel(config["logging"]["log_level"])
        self.planner_log.addHandler(ch)
        self.controller_log = logging.getLogger("Controller")
        self.controller_log.setLevel(config["logging"]["log_level"])
        self.controller_log.addHandler(ch)

        # TODO: How to organize logger for decentralized settings
        # init logger
        self.logger = DataLogger(config)

        self.logger.add("sim_timestep", config["simulation"]["timestep"])
        self.logger.add("duration", config["simulation"]["duration"])

        self.logger.add("nq", ctrl_config["robot"]["dims"]["q"])
        self.logger.add("nv", ctrl_config["robot"]["dims"]["v"])
        self.logger.add("nx", ctrl_config["robot"]["dims"]["x"])
        self.logger.add("nu", ctrl_config["robot"]["dims"]["u"])

        # ROS Related
        self.robot_interface = MobileManipulatorROSInterface()
        self.vicon_tool_interface = ViconObjectInterface("ThingWoodTray")
        # self.vicon_tool_interface = ViconObjectInterface("tool")

        rospy.on_shutdown(self.shutdownhook)
        self.ctrl_c = False
        self.run()

    def shutdownhook(self):
        self.ctrl_c = True
        timestamp = datetime.datetime.now()
        self.robot_interface.brake()
        self.logger.save(timestamp, "control")

    def run(self):
        rate = rospy.Rate(self.ctrl_rate)


        while not self.robot_interface.ready() or not self.vicon_tool_interface.ready():
            self.robot_interface.brake()
            rate.sleep()

            if rospy.is_shutdown():
                return

        print("Controller received joint states. Proceed ... ")
        self.planner_coord_transform(self.robot_interface.q, self.planner_config)
        self.sot = SoTStatic(self.planner_config)

        print("robot coord: {}".format(self.robot_interface.q))
        for planner in self.sot.planners:
            print("planner target:{}".format(planner.getTrackingPoint(0)))

        input("Press Enter to continue...")
        t = rospy.Time.now().to_sec()
        t0 = t

        while not self.ctrl_c:
            t1 = rospy.Time.now().to_sec()
            if t1 - t > (1./ self.ctrl_rate)*5:
                self.controller_log.debug("Controller running slow. Last interval {}".format(t1 -t))
            t = t1

            # open-loop command
            robot_states = (self.robot_interface.q, self.robot_interface.v)
            # print("Msg Oldness Base: {}s, Arm: {}s".format(t - robot_interface.base.last_msg_time, t - robot_interface.arm.last_msg_time))
            # print("q: {}, v:{}, u: {}, acc:{}".format(robot_states[0][0], robot_states[1][0], u[0], acc))
            # tc1 = time.perf_counter()
            planners = self.sot.getPlanners(num_planners=2)
            tc1_ros = rospy.Time.now().to_sec()
            u, acc = self.controller.control(t-t0, robot_states, planners)
            tc2_ros = rospy.Time.now().to_sec()
            # print("Controller Time (ROS): {}s ".format(tc2_ros - tc1_ros))
            # tc2 = time.perf_counter()
            # print(tc2 - tc1)

            self.robot_interface.publish_cmd_vel(u)

            # Update Task Manager
            states = {"base": robot_states[0][:3], "EE": (self.vicon_tool_interface.position, self.vicon_tool_interface.orientation)}
            self.sot.update(t, states)
            # log
            self.logger.append("ts", t)
            self.log_mpc_info(self.logger, self.controller)
            self.logger.append("controller_run_time", tc2_ros - tc1_ros)
            r_ew_wd = []
            r_bw_wd = []
            for planner in planners:
                if planner.type == "EE":
                    r_ew_wd, _ = planner.getTrackingPoint(t, robot_states)
                elif planner.type == "base":
                    r_bw_wd, _ = planner.getTrackingPoint(t, robot_states)
            if len(r_ew_wd) > 0:
                self.logger.append("r_ew_w_ds", r_ew_wd)
            if len(r_bw_wd) > 0:
                self.logger.append("r_bw_w_ds", r_bw_wd)
            self.logger.append("cmd_vels", u)

            rate.sleep()

        # robot_interface.brake()
        # timestamp = datetime.datetime.now()
        # logger.save(timestamp, "ctrl")

    def log_mpc_info(self, logger, controller):
        logger.append("mpc_solver_statuss", controller.solver_status)
        logger.append("mpc_cost_iters", controller.cost_iter)
        logger.append("mpc_cost_finals", controller.cost_final)
        logger.append("mpc_step_sizes", controller.step_size)

    def planner_coord_transform(self, q, planner_config):
        R_wb = rotz(q[2])
        for task in planner_config["tasks"]:
            if task["planner_type"] == "EESimplePlanner":
                task["target_pos"] = R_wb @ task["target_pos"] + np.hstack((q[:2],0))
            elif task["planner_type"] == "BaseSingleWaypoint":
                task["target_pos"] = (R_wb @ np.hstack((task["target_pos"], 1)))[:2] + q[:2]

if __name__ == "__main__":
    rospy.init_node("controller_ros")

    node = ControllerROSNode()
    try:
        node.run()
    except rospy.ROSInterruptException:
        node.robot_interface.brake()
        timestamp = datetime.datetime.now()
        node.logger.save(timestamp, "ctrl")
