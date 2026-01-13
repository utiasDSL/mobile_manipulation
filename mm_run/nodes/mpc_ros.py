import argparse
import copy
import logging
import os
import sys
import threading
import time

import numpy as np
import rospy
import tf.transformations as tf
from geometry_msgs.msg import Point, PoseStamped, Quaternion, Transform, Twist
from mobile_manipulation_central import PointToPointTrajectory, bound_array
from mobile_manipulation_central.ros_interface import (
    JoystickButtonInterface,
    MobileManipulatorROSInterface,
    ViconObjectInterface,
)
from nav_msgs.msg import Path
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as Rot
from spatialmath.base import r2q, rpy2r
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

import mm_control.MPC as MPC
from mm_control.robot import CasadiModelInterface, MobileManipulator3D
from mm_plan.TaskManager import TaskManager
from mm_utils import parsing
from mm_utils.enums import RefType
from mm_utils.logging import DataLogger
from mm_utils.math import wrap_pi_scalar


class ControllerROSNode:
    def __init__(self):
        np.set_printoptions(precision=3, suppress=True)
        argv = rospy.myargv(argv=sys.argv)
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-c", "--config", required=True, help="Path to configuration file."
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
        print(self.ctrl_config["type"])
        # controller
        control_class = getattr(MPC, self.ctrl_config["type"], None)
        if control_class is None:
            raise ValueError(f"Unknown controller type: {self.ctrl_config['type']}")

        self.controller = control_class(self.ctrl_config)

        self.ctrl_rate = self.ctrl_config["ctrl_rate"]
        self.cmd_vel_pub_rate = self.ctrl_config["cmd_vel_pub_rate"]
        self.mpc_dt = self.ctrl_config["dt"]

        # set py logger level
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        self.planner_log = logging.getLogger("Planner")
        self.planner_log.setLevel(config["logging"]["log_level"])
        self.planner_log.addHandler(ch)
        self.controller_log = logging.getLogger("Controller")
        self.controller_log.setLevel(config["logging"]["log_level"])
        self.controller_log.addHandler(ch)

        # init logger
        self.logger = DataLogger(copy.deepcopy(config), name="control")

        self.logger.add("sim_timestep", config["simulation"]["timestep"])
        self.logger.add("duration", config["simulation"]["duration"])

        self.logger.add("nq", self.ctrl_config["robot"]["dims"]["q"])
        self.logger.add("nv", self.ctrl_config["robot"]["dims"]["v"])
        self.logger.add("nx", self.ctrl_config["robot"]["dims"]["x"])
        self.logger.add("nu", self.ctrl_config["robot"]["dims"]["u"])

        # Get shared timestamp from ROS parameter (set by sim node)
        # Wait for sim node to set it
        while not rospy.has_param("/experiment_timestamp"):
            rospy.sleep(0.1)
        self.session_timestamp = rospy.get_param("/experiment_timestamp")

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

        casadi_kin_dyn = CasadiModelInterface(self.ctrl_config)
        if self.ctrl_config["self_collision_emergency_stop"]:
            self.self_collision_func = casadi_kin_dyn.signedDistanceSymMdlsPerGroup[
                "self"
            ]
            self.ground_collision_func = casadi_kin_dyn.signedDistanceSymMdlsPerGroup[
                "static_obstacles"
            ]["ground"]

        self.controller_visualization_pub = rospy.Publisher(
            "controller_visualization", Marker, queue_size=10
        )
        self.controller_visualization_array_pub = rospy.Publisher(
            "controller_visualization_array", MarkerArray, queue_size=10
        )
        self.plan_visualization_pub = rospy.Publisher(
            "plan_visualization", Marker, queue_size=10
        )
        self.pose_plan_visualization_pub = rospy.Publisher(
            "pose_plan_visualization", PoseStamped, queue_size=10
        )

        self.current_plan_visualization_pub = rospy.Publisher(
            "current_plan_visualization", Marker, queue_size=10
        )
        self.controller_ref_pub = rospy.Publisher(
            "controller_reference", Path, queue_size=5
        )

        self.tracking_point_pub = rospy.Publisher(
            "controller_tracking_pt", MultiDOFJointTrajectory, queue_size=5
        )

        # publish mpc predicted input trajectory at a higher rate
        self.cmd_vel = np.zeros(9)
        self.mpc_plan = None
        self.mpc_plan_time_stamp = 0
        dt_pub = 1.0 / self.cmd_vel_pub_rate
        dt_pub_sec = int(dt_pub)
        dt_pub_nsec = int((dt_pub - dt_pub_sec) * 1e9)
        if self.ctrl_config["cmd_vel_type"] == "integration":
            self.cmd_vel_timer = rospy.Timer(
                rospy.Duration(dt_pub_sec, dt_pub_nsec), self._publish_cmd_vel
            )
        elif self.ctrl_config["cmd_vel_type"] == "interpolation":
            self.cmd_vel_timer = rospy.Timer(
                rospy.Duration(dt_pub_sec, dt_pub_nsec), self._publish_cmd_vel_new
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

    def _publish_cmd_vel(self, event):
        if self.mpc_plan is not None:
            t = rospy.Time.now().to_sec()

            self.lock.acquire()
            t_elasped = t - self.mpc_plan_time_stamp
            self.cmd_vel += (
                self.mpc_plan_interp(t_elasped)
                * (event.current_real - event.last_real).to_sec()
            )

            self.lock.release()

        self.robot_interface.publish_cmd_vel(self.cmd_vel)

    def _publish_cmd_vel_new(self, event):
        if self.mpc_plan is not None:
            t = rospy.Time.now().to_sec()
            self.lock.acquire()
            t_elasped = t - self.mpc_plan_time_stamp
            self.cmd_vel = self.mpc_plan_interp(t_elasped)
            self.lock.release()

        self.robot_interface.publish_cmd_vel(self.cmd_vel)

    def _publish_trajectory_tracking_pt(self, t, robot_states, planner):
        msg = MultiDOFJointTrajectory()
        msg.header.stamp = rospy.Time.now()

        # Get base reference if available
        if planner.has_base_ref:
            p, v = planner.getBaseTrackingPoint(t, robot_states)
            if p is not None:
                msg.joint_names.append("base")
                pt_msg = MultiDOFJointTrajectoryPoint()
                transform = Transform()
                transform.translation.x = p[0]
                transform.translation.y = p[1]
                transform.translation.z = 0.25  # Display height
                quat = tf.quaternion_from_euler(0, 0, p[2])
                transform.rotation.x = quat[0]
                transform.rotation.y = quat[1]
                transform.rotation.z = quat[2]
                transform.rotation.w = quat[3]
                pt_msg.transforms.append(transform)
                if v is not None:
                    velocity = Twist()
                    velocity.linear.x = v[0]
                    velocity.linear.y = v[1]
                    velocity.angular.z = v[2]
                    pt_msg.velocities.append(velocity)
                msg.points.append(pt_msg)

        # Get EE reference if available
        if planner.has_ee_ref:
            p, v = planner.getEETrackingPoint(t, robot_states)
            if p is not None:
                msg.joint_names.append("EE")
                pt_msg = MultiDOFJointTrajectoryPoint()
                transform = Transform()
                transform.translation.x = p[0]
                transform.translation.y = p[1]
                transform.translation.z = p[2]
                quat = tf.quaternion_from_euler(*p[3:])
                transform.rotation.x = quat[0]
                transform.rotation.y = quat[1]
                transform.rotation.z = quat[2]
                transform.rotation.w = quat[3]
                pt_msg.transforms.append(transform)
                if v is not None:
                    velocity = Twist()
                    velocity.linear.x = v[0]
                    velocity.linear.y = v[1]
                    velocity.linear.z = v[2]
                    velocity.angular.x = v[3] if len(v) > 3 else 0
                    velocity.angular.y = v[4] if len(v) > 4 else 0
                    velocity.angular.z = v[5] if len(v) > 5 else 0
                    pt_msg.velocities.append(velocity)
                msg.points.append(pt_msg)

        if len(msg.points) > 0:
            self.tracking_point_pub.publish(msg)

    def _publish_controller_reference(self, ref_pose, ref_velocity):
        # send reference poses as a path
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "world"
        for i in range(len(ref_pose)):
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = "world"
            pose.pose.position.x = ref_pose[i][0]
            pose.pose.position.y = ref_pose[i][1]
            quat = tf.quaternion_from_euler(0, 0, ref_pose[i][2])
            pose.pose.orientation.x = 0
            pose.pose.orientation.y = 0
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            path_msg.poses.append(pose)
        self.controller_ref_pub.publish(path_msg)

    def _make_marker(self, marker_type, id, rgba, scale):
        # make a visualization marker array for the occupancy grid
        m = Marker()
        m.header.frame_id = "world"
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
        m.lifetime = rospy.Duration.from_sec(1.0 / self.ctrl_rate)

        m.pose.orientation.w = 1

        return m

    def _publish_planner_data(self, event):
        self.sot_lock.acquire()
        for pid, planner in enumerate(self.sot.planners):
            color = [0] * 3
            color[pid % 3] = 1

            # Visualize base waypoint/path if available
            if planner.has_base_ref:
                if planner.ref_type == RefType.WAYPOINT:
                    quat = tf.quaternion_from_euler(0, 0, planner.base_target[2])
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = rospy.Time()
                    pose_msg.pose.position = Point(*planner.base_target[:2], 0.25)
                    pose_msg.pose.orientation = Quaternion(*list(quat))
                    self.pose_plan_visualization_pub.publish(pose_msg)
                elif planner.ref_type == RefType.PATH:
                    marker_plan = self._make_marker(
                        Marker.POINTS, pid, rgba=color + [1], scale=[0.1, 0.1, 0.1]
                    )
                    marker_plan.points = [
                        Point(*pt[:2], 0) for pt in planner.base_plan["p"]
                    ]
                    marker_plan.lifetime = rospy.Duration.from_sec(0.1)
                    self.plan_visualization_pub.publish(marker_plan)

            # Visualize EE waypoint/path if available
            if planner.has_ee_ref:
                if planner.ref_type == RefType.WAYPOINT:
                    quat = tf.quaternion_from_euler(*planner.ee_target[3:])
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = rospy.Time()
                    pose_msg.header.frame_id = "world"
                    pose_msg.pose.position = Point(*planner.ee_target[:3])
                    pose_msg.pose.orientation = Quaternion(*list(quat))
                    self.pose_plan_visualization_pub.publish(pose_msg)
                elif planner.ref_type == RefType.PATH:
                    marker_plan = self._make_marker(
                        Marker.POINTS,
                        pid + 100,
                        rgba=color + [1],
                        scale=[0.1, 0.1, 0.1],
                    )
                    marker_plan.points = [Point(*pt[:3]) for pt in planner.ee_plan["p"]]
                    marker_plan.lifetime = rospy.Duration.from_sec(0.1)
                    self.plan_visualization_pub.publish(marker_plan)

        planner = self.sot.getPlanner()
        # Visualize current base waypoint/path if available
        if planner.has_base_ref:
            if planner.ref_type == RefType.WAYPOINT:
                quat = tf.quaternion_from_euler(0, 0, planner.base_target[2])
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time()
                pose_msg.pose.position = Point(*planner.base_target[:2], 0.25)
                pose_msg.pose.orientation = Quaternion(*list(quat))
                self.pose_plan_visualization_pub.publish(pose_msg)
            elif planner.ref_type == RefType.PATH:
                marker_plan = self._make_marker(
                    Marker.POINTS,
                    0,
                    rgba=[1, 0, 0, 1],
                    scale=[0.1, 0.1, 0.1],
                )
                marker_plan.points = [
                    Point(*pt[:2], 0) for pt in planner.base_plan["p"]
                ]
                marker_plan.lifetime = rospy.Duration.from_sec(0.1)
                self.current_plan_visualization_pub.publish(marker_plan)

        # Visualize current EE waypoint/path if available
        if planner.has_ee_ref:
            if planner.ref_type == RefType.WAYPOINT:
                quat = tf.quaternion_from_euler(*planner.ee_target[3:])
                pose_msg = PoseStamped()
                pose_msg.header.frame_id = "world"
                pose_msg.header.stamp = rospy.Time()
                pose_msg.pose.position = Point(*planner.ee_target[:3])
                pose_msg.pose.orientation = Quaternion(*list(quat))
                self.pose_plan_visualization_pub.publish(pose_msg)
            elif planner.ref_type == RefType.PATH:
                marker_plan = self._make_marker(
                    Marker.POINTS,
                    100,
                    rgba=[0, 1, 0, 1],
                    scale=[0.1, 0.1, 0.1],
                )
                marker_plan.points = [Point(*pt[:3]) for pt in planner.ee_plan["p"]]
                marker_plan.lifetime = rospy.Duration.from_sec(0.1)
                self.current_plan_visualization_pub.publish(marker_plan)

        self.sot_lock.release()

    def _publish_mpc_data(self, controller):
        # ee prediction
        marker_ee = self._make_marker(
            Marker.POINTS, 0, rgba=[1.0, 1.0, 1.0, 1], scale=[0.1, 0.1, 0.1]
        )
        marker_ee.points = [Point(*pt) for pt in controller.ee_bar]
        self.controller_visualization_pub.publish(marker_ee)

        # base prediction
        marker_base = self._make_marker(
            Marker.POINTS, 1, rgba=[1.0, 1.0, 1.0, 1], scale=[0.1, 0.1, 0.1]
        )
        marker_base.points = [Point(*pt[:2], 0) for pt in controller.base_bar]
        self.controller_visualization_pub.publish(marker_base)

        # ee tracking points
        if len(controller.ree_bar) > 0 and controller.ree_bar[0].shape[0] == 3:
            marker_ree = self._make_marker(
                Marker.POINTS, 2, rgba=[0.0, 1.0, 1.0, 1], scale=[0.1, 0.1, 0.1]
            )
            marker_ree.points = [Point(*pt[:3]) for pt in controller.ree_bar]
            self.controller_visualization_pub.publish(marker_ree)

        # base tracking points
        marker_rbase = self._make_marker(
            Marker.POINTS, 3, rgba=[0.0, 0.0, 1, 1], scale=[0.1] * 3
        )
        marker_rbase.points = [Point(*pt[:2], 0) for pt in controller.rbase_bar]
        self.controller_visualization_pub.publish(marker_rbase)

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

        states = (self.robot_interface.q, self.robot_interface.v)
        print(f"robot coord: {self.robot_interface.q}")
        self.sot = TaskManager(self.planner_config.copy())

        print("-----Checking Planners----- ")
        for planner in self.sot.planners:
            while not planner.ready():
                self.robot_interface.brake()
                rate.sleep()

                if rospy.is_shutdown():
                    return
            # Print planner targets
            targets = []
            if planner.has_base_ref:
                if planner.ref_type == RefType.WAYPOINT:
                    targets.append(f"base: {planner.base_target}")
                else:
                    targets.append(
                        f"base: path with {len(planner.base_plan['p'])} points"
                    )
            if planner.has_ee_ref:
                if planner.ref_type == RefType.WAYPOINT:
                    targets.append(f"EE: {planner.ee_target}")
                else:
                    targets.append(f"EE: path with {len(planner.ee_plan['p'])} points")
            print(f"planner {planner.name} targets: {', '.join(targets)}")

        print("-----Checking Vicon Tool messages----- ")
        use_vicon_tool_data = True
        if not self.vicon_tool_interface.ready():
            use_vicon_tool_data = False
            print(
                "Controller did not receive vicon tool "
                + self.ctrl_config["robot"]["tool_vicon_name"]
                + ". Using Robot Model"
            )
            self.robot = MobileManipulator3D(self.ctrl_config)
        else:
            print(
                "Controller received vicon tool "
                + self.ctrl_config["robot"]["tool_vicon_name"]
            )

        print("-----Checking Joy stick messages----- ")
        if self.use_joy:
            if self.joystick_interface.ready():
                print("Received joystick msg. Using joystick data.")
            else:
                self.use_joy = False
                print("Did not receive joystick msg.")

        rospy.Timer(rospy.Duration(0, int(1e8)), self._publish_planner_data)

        if self.use_joy:
            print("----- Press Square(Ps4) to start -----")
            while not self.start_end_button_interface.button == 1:
                rate.sleep()

            self.start_end_button_interface.reset_button()
        else:
            input("----- Press Enter to start -----")

        self.sot.activatePlanners()
        t = rospy.Time.now().to_sec()
        t0 = t
        self.sot.started = True

        while not self.ctrl_c:
            t = rospy.Time.now().to_sec()

            # open-loop command
            robot_states = (self.robot_interface.q, self.robot_interface.v)
            # check collision
            q = robot_states[0]
            if self.ctrl_config["self_collision_emergency_stop"]:
                signed_dist_self = self.self_collision_func(q).full().flatten()
                signed_dist_ground = self.ground_collision_func(q).full().flatten()

                if min(signed_dist_self) < 0.05 or min(signed_dist_ground) < 0.05:
                    self.controller_log.warning("Self Collision Detected. Braking!!!!")
                    self.cmd_vel_timer.shutdown()

                    self.robot_interface.brake()
                    continue

            # Get references from TaskManager
            self.sot_lock.acquire()
            references = self.sot.getReferences(
                t - t0, robot_states, self.controller.N + 1, self.controller.dt
            )
            self.sot_lock.release()

            tc1 = time.perf_counter()
            v_bar, u_bar = self.controller.control(t - t0, robot_states, references)
            tc2 = time.perf_counter()
            self.controller_log.log(20, f"Controller Run Time: {tc2 - tc1}")

            # if the robot is very close to the goal, stop the robot
            # Check if any active planner is close to finish
            self.sot_lock.acquire()
            planner = self.sot.getPlanner()
            close_to_goal = planner.closeToFinish()
            self.sot_lock.release()
            if close_to_goal:
                print("Close to goal. Braking")
                v_bar[:, :3] = 0

            if self.ctrl_config["cmd_vel_type"] == "interpolation":
                mpc_plan = v_bar
                N = mpc_plan.shape[0]
                t_mpc = np.arange(N) * self.mpc_dt
                mpc_plan_interp = interp1d(
                    t_mpc,
                    mpc_plan,
                    axis=0,
                    bounds_error=False,
                    fill_value="extrapolate",
                )
            elif self.ctrl_config["cmd_vel_type"] == "integration":
                mpc_plan = u_bar
                N = mpc_plan.shape[0]
                t_mpc = np.arange(N) * self.mpc_dt
                mpc_plan_interp = interp1d(
                    t_mpc,
                    mpc_plan,
                    axis=0,
                    bounds_error=False,
                    fill_value=np.zeros_like(u_bar[0]),
                )
            self.lock.acquire()
            self.mpc_plan = mpc_plan
            self.mpc_plan_time_stamp = t
            self.mpc_plan_interp = mpc_plan_interp
            self.lock.release()

            # publish data
            self._publish_mpc_data(self.controller)
            # Get active planner for visualization
            self.sot_lock.acquire()
            active_planner = self.sot.getPlanner()
            self.sot_lock.release()
            self._publish_trajectory_tracking_pt(t - t0, robot_states, active_planner)

            # Update Task Manager
            # Convert to pose arrays in world frame
            if use_vicon_tool_data:
                ee_pos = self.vicon_tool_interface.position
                ee_quat = self.vicon_tool_interface.orientation
                # For Vicon data, velocity is not directly available, set to zeros
                ee_vel = np.zeros(6)
            else:
                ee_pos, ee_quat = self.robot.getEE(robot_states[0])
                # Compute EE velocity using spatial Jacobian if available
                tool_name = self.robot.tool_link_name
                spatial_jac_key = tool_name + "_spatial"
                if spatial_jac_key in self.robot.jacSymMdls:
                    J_spatial = self.robot.jacSymMdls[spatial_jac_key](robot_states[0])
                    ee_vel = (J_spatial @ robot_states[1]).toarray().flatten()
                else:
                    # Fallback: use position Jacobian and pad with zeros for angular velocity
                    J_pos = self.robot.jacSymMdls[tool_name](robot_states[0])
                    ee_lin_vel = (J_pos @ robot_states[1]).toarray().flatten()
                    ee_vel = np.hstack([ee_lin_vel, np.zeros(3)])

            ee_euler = Rot.from_quat(ee_quat).as_euler("xyz")
            ee_pose = np.hstack([ee_pos, ee_euler])

            base_pose = robot_states[0][:3]  # [x, y, yaw] already in world frame
            base_vel = robot_states[1][:3]  # [vx, vy, vyaw]

            states = {
                "base": {"pose": base_pose, "velocity": base_vel},
                "EE": {"pose": ee_pose, "velocity": ee_vel},
            }
            if self.use_joy:
                self.joystick_interface.button_lock.acquire()
                button = self.joystick_interface.button
                self.joystick_interface.button_lock.release()
                states["joy"] = button

            self.sot_lock.acquire()
            updated, _ = self.sot.update(t - t0, states)
            self.sot_lock.release()

            if self.use_joy and updated:
                self.joystick_interface.reset_button()

            # log
            self.logger.append("ts", t)
            self.log_mpc_info(self.logger, self.controller)
            self.logger.append("controller_run_time", tc2 - tc1)
            r_ew_wd = None
            r_bw_wd = None
            v_ew_wd = None
            v_bw_wd = None
            Q_we_d = None
            ω_ew_wd = None

            # Extract reference data from references dictionary for logging
            if references.get("ee_pose") is not None:
                ee_ref = references["ee_pose"][0]  # Current reference
                r_ew_wd = ee_ref[:3]
                Q_we_d = r2q(rpy2r(ee_ref[3:]), order="xyzs")
                if references.get("ee_velocity") is not None:
                    ee_vel_ref = references["ee_velocity"][0]
                    v_ew_wd = ee_vel_ref[:3]
                    ω_ew_wd = ee_vel_ref[3:]

            if references.get("base_pose") is not None:
                base_ref = references["base_pose"][0]  # Current reference
                r_bw_wd = base_ref
                if references.get("base_velocity") is not None:
                    v_bw_wd = references["base_velocity"][0]
            if r_ew_wd is not None:
                self.logger.append("r_ew_w_ds", r_ew_wd)
            if v_ew_wd is not None:
                self.logger.append("v_ew_w_ds", v_ew_wd)
            if Q_we_d is not None:
                self.logger.append("Q_we_ds", Q_we_d)
            if ω_ew_wd is not None:
                self.logger.append("ω_ew_wds", ω_ew_wd)

            if r_bw_wd is not None:
                if r_bw_wd.shape[0] == 2:
                    self.logger.append("r_bw_w_ds", r_bw_wd)

                elif r_bw_wd.shape[0] == 3:
                    self.logger.append("r_bw_w_ds", r_bw_wd[:2])
                    self.logger.append("yaw_bw_w_ds", r_bw_wd[2])
            if v_bw_wd is not None:
                if v_bw_wd.shape[0] == 2:
                    self.logger.append("v_bw_w_ds", v_bw_wd)
                elif v_bw_wd.shape[0] == 3:
                    self.logger.append("v_bw_w_ds", v_bw_wd[:2])
                    self.logger.append("ω_bw_w_ds", v_bw_wd[2])

            if self.use_joy and self.start_end_button_interface.button == 1:
                break

            rate.sleep()

        self.cmd_vel_timer.shutdown()
        self.mpc_plan = None

        # self.go_home()

    def go_home(self):
        rate = rospy.Rate(125)
        q = self.robot_interface.q
        q[2] = wrap_pi_scalar(q[2])
        print(q)

        trajectory = PointToPointTrajectory.quintic(
            q, self.home, max_vel=0.2, max_acc=1, min_duration=1
        )

        # use P control + feedforward velocity to track the trajectory
        while not rospy.is_shutdown():
            q = self.robot_interface.q
            q[2] = wrap_pi_scalar(q[2])

            t = rospy.Time.now().to_sec()

            dist = np.linalg.norm(self.home - q)

            # we want to both let the trajectory complete and ensure we've
            # converged properly
            if trajectory.done(t) and dist < 1e-2:
                break

            qd, vd, _ = trajectory.sample(t)
            cmd_vel = (qd - q) + vd

            # this shouldn't be needed unless the trajectory is poorly tracked, but
            # we do it just in case for safety
            cmd_vel = bound_array(cmd_vel, lb=-0.2, ub=0.2)

            self.robot_interface.publish_cmd_vel(cmd_vel, bodyframe=False)

            rate.sleep()

        self.robot_interface.brake()

        print(f"Converged to within {dist} of home position.")

    def log_mpc_info(self, logger, controller):
        for key, val in controller.log.items():
            logger.append("_".join(["mpc", key]) + "s", val)


if __name__ == "__main__":
    rospy.init_node("controller_ros")

    node = ControllerROSNode()
    node.run()
