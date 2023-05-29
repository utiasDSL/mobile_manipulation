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
from mmseq_control.IDKC import IKCPrioritized
from mmseq_simulator import simulation
import mmseq_plan.TaskManager as TaskManager
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
        elif ctrl_config["type"] == "TP-IDKC":
            self.controller = IKCPrioritized(ctrl_config)

        self.ctrl_rate = ctrl_config["ctrl_rate"]

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
        # self.vicon_tool_interface = ViconObjectInterface("ThingWoodTray")
        self.vicon_tool_interface = ViconObjectInterface(ctrl_config["robot"]["tool_vicon_name"])

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
        planner_class = getattr(TaskManager, self.planner_config["sot_type"])
        self.sot = planner_class(self.planner_config)
        self.planner_coord_transform(self.robot_interface.q, self.vicon_tool_interface.position, self.sot.planners)

        print("robot coord: {}".format(self.robot_interface.q))
        for planner in self.sot.planners:
            print("planner target:{}".format(planner.getTrackingPoint(0)))

        input("Press Enter to continue...")
        t = rospy.Time.now().to_sec()
        t0 = t
        t_prev = t
        u_prev = 0
        while not self.ctrl_c:
            t = rospy.Time.now().to_sec()

            # open-loop command
            robot_states = (self.robot_interface.q, self.robot_interface.v)
            planners = self.sot.getPlanners(num_planners=2)
            tc1 = time.perf_counter()
            u, acc = self.controller.control(t-t0, robot_states, planners)
            tc2 = time.perf_counter()
            self.controller_log.log(5, "Controller Run Time: {}".format(tc2 - tc1))

            self.robot_interface.publish_cmd_vel(u)

            # Update Task Manager
            states = {"base": robot_states[0][:3], "EE": (self.vicon_tool_interface.position, self.vicon_tool_interface.orientation)}
            self.sot.update(t-t0, states)
            # log
            self.logger.append("ts", t)
            self.logger.append("controller_run_time", tc2 - tc1)
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
            if t - t_prev != 0:
                self.logger.append("cmd_accs", (u - u_prev) / (t - t_prev))
            else:
                self.logger.append("cmd_accs", (u - u_prev) / self.ctrl_rate)

            u_prev = u
            t_prev = t

            rate.sleep()

        # robot_interface.brake()
        # timestamp = datetime.datetime.now()
        # logger.save(timestamp, "ctrl")

    def planner_coord_transform(self, q, ree, planners):
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
                planner.plan = planner.plan @ R_wb.T + P

            elif planner.__class__.__name__ == "BaseSingleWaypoint":
                planner.target_pos = (R_wb @ np.hstack((planner.target_pos, 0)))[:2] + P[:2]
            elif planner.__class__.__name__ == "BasePosTrajectoryCircle":
                planner.c = R_wb[:2, :2] @ planner.c + P[:2]
                planner.plan = planner.plan @ R_wb[:2, :2].T + P[:2]
            elif planner.__class__.__name__ == "BasePosTrajectoryLine":
                planner.plan = planner.plan @ R_wb[:2, :2].T + P[:2]

if __name__ == "__main__":
    rospy.init_node("controller_ros")

    node = ControllerROSNode()
    try:
        node.run()
    except rospy.ROSInterruptException:
        node.robot_interface.brake()
        timestamp = datetime.datetime.now()
        node.logger.save(timestamp, "ctrl")
