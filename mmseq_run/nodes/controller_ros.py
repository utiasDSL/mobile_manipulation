#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import datetime
import logging
import time
import sys
import threading
import numpy as np
import rospy
from spatialmath.base import rotz

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Transform, Twist
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

import mmseq_control.HTIDKC as HTIDKC
from mmseq_control.robot import MobileManipulator3D
import mmseq_plan.TaskManager as TaskManager
from mmseq_utils import parsing
from mmseq_utils.logging import DataLogger
from mobile_manipulation_central.ros_interface import MobileManipulatorROSInterface, ViconObjectInterface

class ControllerROSNode:

    def __init__(self):

        np.set_printoptions(precision=3, suppress=True)
        argv = rospy.myargv(argv=sys.argv)
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True, help="Path to configuration file.")
        parser.add_argument("--ctrl_config", type=str,
                            help="controller config. This overwrites the yaml settings in config if not set to default")
        parser.add_argument("--planner_config", type=str,
                            help="plannner config. This overwrites the yaml settings in config if not set to default")
        parser.add_argument("--logging_sub_folder", type=str,
                            help="save data in a sub folder of logging director")
        args = parser.parse_args(argv[1:])

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

        self.ctrl_config = config["controller"]
        self.planner_config = config["planner"]

        # controller
        control_class = getattr(HTIDKC, self.ctrl_config["type"], None)
        self.controller = control_class(self.ctrl_config)
        self.ctrl_rate = self.ctrl_config["ctrl_rate"]

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

        self.logger.add("nq", self.ctrl_config["robot"]["dims"]["q"])
        self.logger.add("nv", self.ctrl_config["robot"]["dims"]["v"])
        self.logger.add("nx", self.ctrl_config["robot"]["dims"]["x"])
        self.logger.add("nu", self.ctrl_config["robot"]["dims"]["u"])

        # ROS Related
        self.robot_interface = MobileManipulatorROSInterface()
        self.vicon_tool_interface = ViconObjectInterface(self.ctrl_config["robot"]["tool_vicon_name"])
        self.plan_visualization_pub = rospy.Publisher("plan_visualization", Marker, queue_size=10)
        self.tracking_point_pub = rospy.Publisher("controller_tracking_pt", MultiDOFJointTrajectory, queue_size=5)

        self.sot_lock = threading.Lock()
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


        while not self.robot_interface.ready():
            self.robot_interface.brake()
            rate.sleep()

            if rospy.is_shutdown():
                return

        print("Controller received joint states. Proceed ... ")

        use_vicon_tool_data = True
        if not self.vicon_tool_interface.ready():
            use_vicon_tool_data = False
            print("Controller did not receive vicon tool " + self.ctrl_config["robot"]["tool_vicon_name"] + ". Using Robot Model")
            self.robot = MobileManipulator3D(self.ctrl_config)
        else:
            print("Controlelr received vicon tool " + self.ctrl_config["robot"]["tool_vicon_name"])

        planner_class = getattr(TaskManager, self.planner_config["sot_type"])
        self.sot = planner_class(self.planner_config)
        if use_vicon_tool_data:
            self.planner_coord_transform(self.robot_interface.q, self.vicon_tool_interface.position, self.sot.planners)
        else:
            ee_pos, _ = self.robot.getEE(self.robot_interface.q)
            self.planner_coord_transform(self.robot_interface.q, ee_pos, self.sot.planners)

        rospy.Timer(rospy.Duration(0, int(1e8)), self._publish_planner_data)

        print("robot coord: {}".format(self.robot_interface.q))
        for planner in self.sot.planners:
            print("planner target:{}".format(planner.getTrackingPoint(0)))

        input("Press Enter to continue...")
        # rospy.sleep(5)
        t = rospy.Time.now().to_sec()
        t0 = t
        while not self.ctrl_c and t - t0 < 16:
            t = rospy.Time.now().to_sec()

            # open-loop command
            robot_states = (self.robot_interface.q, self.robot_interface.v)
            self.sot_lock.acquire()
            planners = self.sot.getPlanners(num_planners=2)
            self.sot_lock.release()
            tc1 = time.perf_counter()
            u, acc = self.controller.control(t-t0, robot_states, planners)
            tc2 = time.perf_counter()
            self.controller_log.log(5, "Controller Run Time: {}".format(tc2 - tc1))

            self.robot_interface.publish_cmd_vel(u)

            # Update Task Manager
            if use_vicon_tool_data:
                ee_states = (self.vicon_tool_interface.position, self.vicon_tool_interface.orientation)
            else:
                ee_states = self.robot.getEE(robot_states[0])
            states = {"base": (robot_states[0][:3], robot_states[1][:3]), "EE": ee_states}
            self.sot_lock.acquire()
            self.sot.update(t-t0, states)
            self.sot_lock.release()

            self._publish_trajectory_tracking_pt(t - t0, robot_states, planners)

            # log
            self.logger.append("ts", t)
            self.logger.append("controller_run_time", tc2 - tc1)
            r_ew_wd = []
            r_bw_wd = []
            for planner in planners:
                if planner.type == "EE":
                    if planner.name == "PartialPlanner":
                        r_ew_wd, ref_v_ee = planner.getTrackingPoint(t, robot_states)
                        logger.append("ref_v_ee", ref_v_ee)
                    else:
                        r_ew_wd, _ = planner.getTrackingPoint(t, robot_states)
                elif planner.type == "base":
                    if planner.name == "PartialPlanner":
                        r_bw_wd, ref_v_base = planner.getTrackingPoint(t, robot_states)
                        ref_q_dot, ref_u = planner.getRefVelandAcc(t)
                        logger.append("ref_v_base", ref_v_base)
                        logger.append("ref_vels", ref_q_dot)
                        logger.append("ref_accs", ref_u)
                    else:
                        r_bw_wd, _ = planner.getTrackingPoint(t, robot_states)
            if len(r_ew_wd) > 0:
                self.logger.append("r_ew_w_ds", r_ew_wd)
            if len(r_bw_wd) > 0:
                self.logger.append("r_bw_w_ds", r_bw_wd)
            self.logger.append("cmd_vels", u)
            self.logger.append("cmd_accs", acc)

            if self.controller.__class__.__name__ == "HTIDKC":
                for id, name in enumerate(self.controller.task_names):
                    self.logger.append("task_violations_" + name, self.controller.ws[id])
                # only log once at the beginning.
                if not self.logger.data.get("task_names"):
                    self.logger.append("task_names", self.controller.task_names)
            rate.sleep()

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
                planner.plan['p'] = planner.plan['p'] @ R_wb.T + P

            elif planner.__class__.__name__ == "BaseSingleWaypoint":
                planner.target_pos = (R_wb @ np.hstack((planner.target_pos, 0)))[:2] + P[:2]
            elif planner.__class__.__name__ == "BasePosTrajectoryCircle":
                planner.c = R_wb[:2, :2] @ planner.c + P[:2]
                planner.plan['p'] = planner.plan['p'] @ R_wb[:2, :2].T + P[:2]
            elif planner.__class__.__name__ == "BasePosTrajectoryLine" or planner.__class__.__name__ == "BasePosTrajectorySqaureWave":
                planner.plan['p'] = planner.plan['p'] @ R_wb[:2, :2].T + P[:2]

    def _make_marker(self, marker_type, id, rgba, scale):
        # make a visualization marker array for the occupancy grid
        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = rospy.Time.now()
        m.id = id
        m.type = marker_type
        m.action = Marker.ADD

        m.scale.x = scale[0]
        m.scale.y = scale[1]
        m.scale.z = scale[2]
        m.color.r = rgba[0]
        m.color.g = rgba[1]
        m.color.b = rgba[2]
        m.color.a = rgba[3]
        m.lifetime = rospy.Duration.from_sec(1./self.ctrl_rate)

        m.pose.orientation.w = 1

        return m

    def _publish_planner_data(self, event):

        self.sot_lock.acquire()
        for pid, planner in enumerate(self.sot.planners):
            color = [0] * 3
            color[pid % 3] = 1
            if planner.ref_type == "waypoint":
                if planner.ref_data_type == "Vec3":
                    marker_plan = self._make_marker(Marker.SPHERE, pid, rgba=color + [1], scale=[0.1, 0.1, 0.1])
                    marker_plan.pose.position = Point(*planner.target_pos)
                elif planner.ref_data_type == "Vec2":
                    marker_plan = self._make_marker(Marker.CYLINDER, pid, rgba=color + [1], scale=[0.1, 0.1, 0.5])
                    marker_plan.pose.position = Point(*planner.target_pos, 0.25)
            elif planner.ref_type == "trajectory":
                marker_plan = self._make_marker(Marker.LINE_STRIP, pid, rgba=color + [1], scale=[0.1, 0.1, 0.1])

                if planner.ref_data_type == "Vec3":
                    marker_plan.points = [Point(*pt) for pt in planner.plan['p']]
                elif planner.ref_data_type == "Vec2":
                    marker_plan.points = [Point(*pt, 0) for pt in planner.plan['p']]

            marker_plan.lifetime = rospy.Duration.from_sec(0.1)
            self.plan_visualization_pub.publish(marker_plan)

        self.sot_lock.release()

    def _publish_trajectory_tracking_pt(self, t, robot_states, planners):

        msg = MultiDOFJointTrajectory()
        msg.header.stamp = rospy.Time.now()

        for planner in planners:
            p, v = planner.getTrackingPoint(t, robot_states)

            msg.joint_names.append(planner.type)
            pt_msg = MultiDOFJointTrajectoryPoint()

            if planner.ref_data_type == "Vec3":
                transform = Transform()
                transform.translation.x = p[0]
                transform.translation.y = p[1]
                transform.translation.z = p[2]

                velocity = Twist()
                velocity.linear.x = v[0]
                velocity.linear.y = v[1]
                velocity.linear.z = v[2]

                pt_msg.transforms.append(transform)
                pt_msg.velocities.append(velocity)
            elif planner.ref_data_type == "Vec2":
                transform = Transform()
                transform.translation.x = p[0]
                transform.translation.y = p[1]

                velocity = Twist()
                velocity.linear.x = v[0]
                velocity.linear.y = v[1]

                pt_msg.transforms.append(transform)
                pt_msg.velocities.append(velocity)

            msg.points.append(pt_msg)

        self.tracking_point_pub.publish(msg)

if __name__ == "__main__":
    rospy.init_node("controller_ros")

    node = ControllerROSNode()
    try:
        node.run()
    except rospy.ROSInterruptException:
        node.robot_interface.brake()
