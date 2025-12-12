import argparse
import datetime
import logging
import os
import sys
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import Point
from mobile_manipulation_central.ros_interface import (
    JoystickButtonInterface,
    MapGridInterface,
    MobileManipulatorROSInterface,
    ViconObjectInterface,
)
from trajectory_msgs.msg import MultiDOFJointTrajectory
from visualization_msgs.msg import Marker, MarkerArray

import mm_plan.TaskManager as TaskManager
from mm_utils import parsing
from mm_utils.logging import DataLogger


class ControllerROSNode:
    def __init__(self):
        print("Testing planner")

        np.set_printoptions(precision=3, suppress=True)
        argv = rospy.myargv(argv=sys.argv)
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config", required=True, help="Path to configuration file."
        )
        parser.add_argument(
            "--ctrl_config",
            type=str,
            help="controller config. This overwrites the yaml settings in config if not set to default",
        )
        parser.add_argument(
            "--planner_config",
            type=str,
            help="planner config. This overwrites the yaml settings in config if not set to default",
        )
        parser.add_argument(
            "--logging_sub_folder",
            type=str,
            help="save data in a sub folder of logging directory",
        )
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
            config["logging"]["log_dir"] = os.path.join(
                config["logging"]["log_dir"], args.logging_sub_folder
            )

        self.ctrl_config = config["controller"]
        self.planner_config = config["planner"].copy()
        print("planner config: ", self.planner_config)

        self.ctrl_rate = self.ctrl_config["ctrl_rate"]

        # set py logger level
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        self.planner_log = logging.getLogger("Planner")
        self.planner_log.setLevel(config["logging"]["log_level"])
        self.planner_log.addHandler(ch)

        # init logger
        self.logger = DataLogger(config.copy(), name="control")

        # Create timestamp for logging
        timestamp = datetime.datetime.now()
        self.session_timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

        # ROS Related
        self.robot_interface = MobileManipulatorROSInterface()
        self.vicon_tool_interface = ViconObjectInterface(
            self.ctrl_config["robot"]["tool_vicon_name"]
        )

        self.start_end_button_interface = JoystickButtonInterface(2)  # square

        if self.planner_config.get("use_joy", False):
            self.use_joy = True
            self.joystick_interface = JoystickButtonInterface(1)  # circle
        else:
            self.use_joy = False

        self.map_interface = MapGridInterface(config=self.ctrl_config)

        self.controller_visualization_pub = rospy.Publisher(
            "controller_visualization", Marker, queue_size=10
        )
        self.controller_visualization_array_pub = rospy.Publisher(
            "controller_visualization_array", MarkerArray, queue_size=10
        )
        self.plan_visualization_pub = rospy.Publisher(
            "plan_visualization", Marker, queue_size=10
        )
        self.current_plan_visualization_pub = rospy.Publisher(
            "current_plan_visualization", Marker, queue_size=10
        )

        self.tracking_point_pub = rospy.Publisher(
            "controller_tracking_pt", MultiDOFJointTrajectory, queue_size=5
        )

        self.lock = threading.Lock()
        self.sot_lock = threading.Lock()

        rospy.on_shutdown(self.shutdownhook)
        self.ctrl_c = False
        self.run()

    def shutdownhook(self):
        self.ctrl_c = True
        self.robot_interface.brake()
        self.logger.save(session_timestamp=self.session_timestamp)

    def _publish_planner_data(self, event):
        self.sot_lock.acquire()
        for pid, planner in enumerate(self.sot.planners):
            color = [0] * 3
            color[pid % 3] = 1
            if planner.ref_type == "waypoint":
                if planner.ref_data_type == "Vec3":
                    marker_plan = self._make_marker(
                        Marker.SPHERE, pid, rgba=color + [1], scale=[0.1, 0.1, 0.1]
                    )
                    marker_plan.pose.position = Point(*planner.target_pos)
                elif planner.ref_data_type == "Vec2":
                    marker_plan = self._make_marker(
                        Marker.CYLINDER, pid, rgba=color + [1], scale=[0.1, 0.1, 0.5]
                    )
                    marker_plan.pose.position = Point(*planner.target_pos, 0.25)
            elif planner.ref_type == "trajectory":
                marker_plan = self._make_marker(
                    Marker.LINE_STRIP, pid, rgba=color + [1], scale=[0.1, 0.1, 0.1]
                )

                if planner.ref_data_type == "Vec3":
                    marker_plan.points = [Point(*pt) for pt in planner.plan["p"]]
                elif planner.ref_data_type == "Vec2":
                    marker_plan.points = [Point(*pt, 0) for pt in planner.plan["p"]]

            marker_plan.lifetime = rospy.Duration.from_sec(0.1)
            self.plan_visualization_pub.publish(marker_plan)

        curr_planners = self.sot.getPlanners(2)
        colors = [[1, 0, 0], [0, 1, 0]]
        for pid, planner in enumerate(curr_planners):
            if planner.ref_type == "waypoint":
                if planner.ref_data_type == "Vec3":
                    marker_plan = self._make_marker(
                        Marker.SPHERE,
                        pid,
                        rgba=colors[pid] + [1],
                        scale=[0.1, 0.1, 0.1],
                    )
                    marker_plan.pose.position = Point(*planner.target_pos)
                elif planner.ref_data_type == "Vec2":
                    marker_plan = self._make_marker(
                        Marker.CYLINDER,
                        pid,
                        rgba=colors[pid] + [1],
                        scale=[0.1, 0.1, 0.5],
                    )
                    marker_plan.pose.position = Point(*planner.target_pos, 0.25)
            elif planner.ref_type == "trajectory":
                marker_plan = self._make_marker(
                    Marker.LINE_STRIP,
                    pid,
                    rgba=colors[pid] + [1],
                    scale=[0.1, 0.1, 0.1],
                )

                if planner.ref_data_type == "Vec3":
                    marker_plan.points = [Point(*pt) for pt in planner.plan["p"]]
                elif planner.ref_data_type == "Vec2":
                    marker_plan.points = [Point(*pt, 0) for pt in planner.plan["p"]]

            marker_plan.lifetime = rospy.Duration.from_sec(0.1)
            self.current_plan_visualization_pub.publish(marker_plan)

        self.sot_lock.release()

    def run(self):
        rate = rospy.Rate(self.ctrl_rate)

        print("-----Checking Robot Interface-----")
        while not self.robot_interface.ready():
            self.robot_interface.brake()
            rate.sleep()

            if rospy.is_shutdown():
                return
        print("Controller received joint states. Proceed ... ")
        self.home = self.robot_interface.q

        task_manager_class = getattr(TaskManager, self.planner_config["sot_type"])
        self.sot = task_manager_class(self.planner_config.copy())

        # rospy.sleep(5.)
        self.sot.activatePlanners()
        rospy.Time.now().to_sec()
        self.sot.started = True

        sot_num_plans = self.ctrl_config.get("task_num", None)
        if sot_num_plans is None:
            sot_num_plans = 1 if self.ctrl_config["type"][:2] == "ST" else 2
        while not self.ctrl_c:
            rospy.Time.now().to_sec()
            if self.ctrl_config["sdf_collision_avoidance_enabled"]:
                tm0 = time.perf_counter()
                status, map = self.map_interface.get_map()
                tm1 = time.perf_counter()

                tm1 - tm0

                if status:
                    print("received map")
                else:
                    print("no map received")

            # open-loop command
            robot_states = (self.robot_interface.q, self.robot_interface.v)
            planners = self.sot.getPlanners(num_planners=sot_num_plans)

            planners[0].updateRobotStates(robot_states)  # need robot pose

            if planners[0].plan is not None:
                print("plan t\n", planners[0].plan["t"].shape, planners[0].plan["t"])
                print("plan p\n", planners[0].plan["p"].shape, planners[0].plan["p"])
                print("plan v\n", planners[0].plan["v"].shape, planners[0].plan["v"])
            else:
                print("no plan")

            states = {"base": (robot_states[0][:3], robot_states[1][:3])}
            print("states", states)

            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("controller_ros")

    node = ControllerROSNode()
    node.run()
