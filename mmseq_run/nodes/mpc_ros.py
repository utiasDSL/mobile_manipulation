#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import datetime
import logging
import time
import threading
import sys
import numpy as np
import rospy
import tf.transformations as tf
from spatialmath.base import rotz
from scipy.interpolate import interp1d
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Transform, Twist, PoseStamped, Quaternion
from nav_msgs.msg import Path
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

import mmseq_control.HTMPC as HTMPC
import mmseq_control.STMPC as STMPC
import mmseq_control_new.MPC as MPC
import mmseq_control_new.HTMPC as HTMPCNew


from mmseq_control.robot import MobileManipulator3D, CasadiModelInterface
import mmseq_plan.TaskManager as TaskManager
from mmseq_utils import parsing
from mmseq_utils.logging import DataLogger
from mmseq_utils.math import wrap_pi_scalar, wrap_to_2_pi_scalar
from mobile_manipulation_central.ros_interface import MobileManipulatorROSInterface, ViconObjectInterface, ViconMarkerSwarmInterface, JoystickButtonInterface, MapInterface, MapInterfaceNew, MapGridInterface
from mobile_manipulation_central import PointToPointTrajectory, bound_array

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

        if config["controller"]["type"] == "HTMPCLex":
            config["controller"]["HT_MaxIntvl"] = 1

        if args.logging_sub_folder != "default":
            config["logging"]["log_dir"] = os.path.join(config["logging"]["log_dir"], args.logging_sub_folder)

        self.ctrl_config = config["controller"]
        self.planner_config = config["planner"].copy()
        print(self.ctrl_config["type"])
        # controller
        control_class = getattr(HTMPC, self.ctrl_config["type"], None)
        if control_class is None:
            control_class = getattr(STMPC, self.ctrl_config["type"], None)
        if control_class is None:
            control_class = getattr(MPC, self.ctrl_config["type"], None)
        if control_class is None:
            control_class = getattr(HTMPCNew, self.ctrl_config["type"], None)

        self.controller = control_class(self.ctrl_config)

        self.ctrl_rate = self.ctrl_config["ctrl_rate"]
        self.cmd_vel_pub_rate = self.ctrl_config["cmd_vel_pub_rate"]
        self.mpc_dt = self.ctrl_config["dt"]

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
        self.logger = DataLogger(config.copy())

        self.logger.add("sim_timestep", config["simulation"]["timestep"])
        self.logger.add("duration", config["simulation"]["duration"])

        self.logger.add("nq", self.ctrl_config["robot"]["dims"]["q"])
        self.logger.add("nv", self.ctrl_config["robot"]["dims"]["v"])
        self.logger.add("nx", self.ctrl_config["robot"]["dims"]["x"])
        self.logger.add("nu", self.ctrl_config["robot"]["dims"]["u"])

        # ROS Related
        self.robot_interface = MobileManipulatorROSInterface()
        self.vicon_tool_interface = ViconObjectInterface(self.ctrl_config["robot"]["tool_vicon_name"])
        if self.planner_config["sot_type"] == "SoTSequentialTasks":
            self.vicon_marker_swarm_interface = ViconMarkerSwarmInterface(self.planner_config["vicon_mark_swarm_estimation_topic_name"])

        self.start_end_button_interface = JoystickButtonInterface(2)    # square

        if self.planner_config.get("use_joy", False):
            self.use_joy = True
            self.joystick_interface = JoystickButtonInterface(1)    # circle
        else:
            self.use_joy = False
        
        self.map_interface = MapGridInterface(config=self.ctrl_config)

        casadi_kin_dyn = CasadiModelInterface(self.ctrl_config)
        if self.ctrl_config["self_collision_emergency_stop"]:
            self.self_collision_func = casadi_kin_dyn.signedDistanceSymMdlsPerGroup["self"]
            self.ground_collision_func = casadi_kin_dyn.signedDistanceSymMdlsPerGroup["static_obstacles"]["ground"]


        self.controller_visualization_pub = rospy.Publisher("controller_visualization", Marker, queue_size=10)
        self.controller_visualization_array_pub = rospy.Publisher("controller_visualization_array", MarkerArray, queue_size=10)
        self.plan_visualization_pub = rospy.Publisher("plan_visualization", Marker, queue_size=10)
        self.current_plan_visualization_pub = rospy.Publisher("current_plan_visualization", Marker, queue_size=10)
        self.controller_ref_pub = rospy.Publisher("controller_reference", Path, queue_size=5)

        self.tracking_point_pub = rospy.Publisher("controller_tracking_pt", MultiDOFJointTrajectory, queue_size=5)

        # publish mpc predicted input trajectory at a higher rate
        self.cmd_vel = np.zeros(9)
        self.mpc_plan = None
        self.mpc_plan_time_stamp = 0
        dt_pub = 1./ self.cmd_vel_pub_rate
        dt_pub_sec = int(dt_pub)
        dt_pub_nsec = int((dt_pub - dt_pub_sec) * 1e9)
        self.cmd_vel_timer = rospy.Timer(rospy.Duration(dt_pub_sec, dt_pub_nsec), self._publish_cmd_vel_new)

        self.lock = threading.Lock()
        self.sot_lock = threading.Lock()

        rospy.on_shutdown(self.shutdownhook)
        self.ctrl_c = False
        self.run()

    def shutdownhook(self):
        self.ctrl_c = True
        timestamp = datetime.datetime.now()
        self.robot_interface.brake()
        self.logger.save(timestamp, "control")

    def _publish_cmd_vel(self, event):
        if self.mpc_plan is not None:
            self.lock.acquire()

            t = rospy.Time.now().to_sec()
            t_elasped = t - self.mpc_plan_time_stamp
            acc_indx = int(t_elasped/self.mpc_dt)
            self.cmd_vel += self.mpc_plan[acc_indx] * (event.current_real - event.last_real).to_sec()

            self.lock.release()

        self.robot_interface.publish_cmd_vel(self.cmd_vel)
    
    def _publish_cmd_vel_new(self, event):
        if self.mpc_plan is not None:
            t = rospy.Time.now().to_sec()
            print("t{}, even{}".format(t, event.current_real.to_sec()))
            # print("cmd vel loopo {}".format(t))
            t_elasped = t - self.mpc_plan_time_stamp
            # print("cmd t elapsed {}".format(t_elasped))
            self.lock.acquire()
            self.cmd_vel = self.mpc_plan_interp(t_elasped)
            self.lock.release()

            print("mpc plan t_elapsed{} start {} pub {}".format(t_elasped, self.mpc_plan, self.cmd_vel))
            for i in range(20):
                print(self.mpc_plan_interp(i*0.1))
        self.robot_interface.publish_cmd_vel(self.cmd_vel)

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
                pt_msg.transforms.append(transform)
                
                if v is not None:
                    velocity = Twist()
                    velocity.linear.x = v[0]
                    velocity.linear.y = v[1]
                    velocity.linear.z = v[2]
                    pt_msg.velocities.append(velocity)
            elif planner.ref_data_type == "Vec2":
                transform = Transform()
                transform.translation.x = p[0]
                transform.translation.y = p[1]
                pt_msg.transforms.append(transform)
                if v is not None:
                    velocity = Twist()
                    velocity.linear.x = v[0]
                    velocity.linear.y = v[1]
                    pt_msg.velocities.append(velocity)

            msg.points.append(pt_msg)

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
                elif planner.ref_data_type == "SE2":
                    marker_plan = self._make_marker(Marker.ARROW, pid, rgba=color + [1], scale=[0.1, 0.1, 0.1])
                    marker_plan.pose.position = Point(*planner.target_pose[:2], 0.25)
                    quat = tf.quaternion_from_euler(0,0,planner.target_pose[2])
                    marker_plan.pose.orientation = Quaternion(*list(quat))
                elif planner.ref_data_type == "Vec2":
                    marker_plan = self._make_marker(Marker.CYLINDER, pid, rgba=color + [1], scale=[0.1, 0.1, 0.5])
                    marker_plan.pose.position = Point(*planner.target_pos, 0.25)
            elif planner.ref_type == "trajectory" or planner.ref_type == "path":
                marker_plan = self._make_marker(Marker.POINTS, pid, rgba=color + [1], scale=[0.1, 0.1, 0.1])

                if planner.ref_data_type == "Vec3":
                    marker_plan.points = [Point(*pt) for pt in planner.plan['p']]
                elif planner.ref_data_type == "Vec2" or planner.ref_data_type == "SE2":
                    marker_plan.points = [Point(*pt[:2], 0) for pt in planner.plan['p']]
            marker_plan.lifetime = rospy.Duration.from_sec(0.1)
            self.plan_visualization_pub.publish(marker_plan)

        curr_planners = self.sot.getPlanners(2)
        colors = [[1,0,0],[0,1,0]]
        for pid, planner in enumerate(curr_planners):
            if planner.ref_type == "waypoint":
                if planner.ref_data_type == "Vec3":
                    marker_plan = self._make_marker(Marker.SPHERE, pid, rgba=colors[pid]+[1], scale=[0.1, 0.1, 0.1])
                    marker_plan.pose.position = Point(*planner.target_pos)
                elif planner.ref_data_type == "SE2":
                    marker_plan = self._make_marker(Marker.ARROW, pid, rgba=color + [1], scale=[0.1, 0.1, 0.1])
                    marker_plan.pose.position = Point(*planner.target_pose[:2], 0.25)
                    quat = tf.quaternion_from_euler(0,0,planner.target_pose[2])
                    marker_plan.pose.orientation = Quaternion(*list(quat))
                elif planner.ref_data_type == "Vec2":
                    marker_plan = self._make_marker(Marker.CYLINDER, pid, rgba=colors[pid]+ [1], scale=[0.1, 0.1, 0.5])
                    marker_plan.pose.position = Point(*planner.target_pos, 0.25)
            elif planner.ref_type == "trajectory" or planner.ref_type == "path":
                marker_plan = self._make_marker(Marker.POINTS, pid, rgba=colors[pid]+ [1], scale=[0.1, 0.1, 0.1])

                if planner.ref_data_type == "Vec3":
                    marker_plan.points = [Point(*pt) for pt in planner.plan['p']]
                elif planner.ref_data_type == "Vec2":
                    marker_plan.points = [Point(*pt, 0) for pt in planner.plan['p']]

            marker_plan.lifetime = rospy.Duration.from_sec(0.1)
            self.current_plan_visualization_pub.publish(marker_plan)


        self.sot_lock.release()

    def _publish_mpc_data(self, controller):
        # ee prediction
        marker_ee = self._make_marker(Marker.POINTS, 0, rgba=[1.0, 1.0, 1.0, 1], scale=[0.1, 0.1, 0.1])
        marker_ee.points = [Point(*pt) for pt in controller.ee_bar]
        self.controller_visualization_pub.publish(marker_ee)

        # base prediction
        marker_base = self._make_marker(Marker.POINTS, 1, rgba=[1.0, 1.0, 1.0, 1], scale=[0.1, 0.1, 0.1])
        marker_base.points = [Point(*pt[:2], 0) for pt in controller.base_bar]
        self.controller_visualization_pub.publish(marker_base)

        # ee tracking points
        marker_ree = self._make_marker(Marker.POINTS, 2, rgba=[0.0, 1.0, 1.0, 1], scale=[0.1, 0.1, 0.1])
        marker_ree.points = [Point(*pt) for pt in controller.ree_bar]
        self.controller_visualization_pub.publish(marker_ree)

        # base tracking points
        marker_rbase = self._make_marker(Marker.POINTS, 3, rgba=[0.0, 0.0, 1, 1], scale=[0.1]*3)
        marker_rbase.points = [Point(*pt[:2], 0) for pt in controller.rbase_bar]
        self.controller_visualization_pub.publish(marker_rbase)

        # base sdf gradients
        if self.ctrl_config["sdf_collision_avoidance_enabled"]:
            marker_array_sdf_grad = MarkerArray()
            marker_id = 4
            for i in range(0, controller.sdf_grad_bar["base"].shape[1], 3):
                grad = controller.sdf_grad_bar["base"][:, i]

                marker_sdf_grad_base = self._make_marker(Marker.ARROW, marker_id + i, rgba=[1.0, 0.0, 0, 1.0], scale=[0.05]*3)
                marker_sdf_grad_base.points.append(Point(*controller.base_bar[i]))
                marker_sdf_grad_base.points.append(Point(*(controller.base_bar[i] + grad)))
                marker_array_sdf_grad.markers.append(marker_sdf_grad_base)
                marker_id += 1


            # ee sdf gradients
            for i in range(0, controller.sdf_grad_bar["EE"].shape[1], 3):
                grad = controller.sdf_grad_bar["EE"][:, i]

                marker_sdf_grad_ee = self._make_marker(Marker.ARROW, marker_id+i, rgba=[1.0, 0.0, 0, 1.0], scale=[0.05]*3)
                marker_sdf_grad_ee.points.append(Point(*(controller.ee_bar[i])))
                marker_sdf_grad_ee.points.append(Point(*(controller.ee_bar[i] + grad)))
                marker_array_sdf_grad.markers.append(marker_sdf_grad_ee)
                marker_id += 1


            self.controller_visualization_array_pub.publish(marker_array_sdf_grad)

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

        states = (self.robot_interface.q ,self.robot_interface.v)
        print("robot coord: {}".format(self.robot_interface.q))
        task_manager_class = getattr(TaskManager, self.planner_config["sot_type"])
        self.sot = task_manager_class(self.planner_config.copy())
        print("-----Checking Planners----- ")
        for planner in self.sot.planners:
            while not planner.ready():
                self.robot_interface.brake()
                rate.sleep()

                if rospy.is_shutdown():
                    return
            print("planner {} target:{}".format(planner.name, planner.getTrackingPoint(0, states)))

        if self.ctrl_config["sdf_collision_avoidance_enabled"]:
            print("-----Checking Map Interface----- ")
            while not self.map_interface.ready():
                self.robot_interface.brake()
                rate.sleep()
                if rospy.is_shutdown():
                    return
            
            _, map_latest = self.map_interface.get_map()
            # self.controller.model_interface.sdf_map.update_map(*map_latest)
            # self.controller.model_interface.sdf_map.vis([-0, 4],[-2,2], [0.2,0.2])
            print("Received Map. Proceed ...")
        else:
            map_latest = None

        print("-----Checking Vicon Tool messages----- ")
        use_vicon_tool_data = True
        if not self.vicon_tool_interface.ready():
            use_vicon_tool_data = False
            print("Controller did not receive vicon tool " + self.ctrl_config["robot"]["tool_vicon_name"] + ". Using Robot Model")
            self.robot = MobileManipulator3D(self.ctrl_config)
        else:
            print("Controller received vicon tool " + self.ctrl_config["robot"]["tool_vicon_name"])


        print("-----Checking Vicon Marker Swarm Estimation messages----- ")
        use_vicon_marker_swarm_data = True
        if self.planner_config["sot_type"] == "SoTSequentialTasks":
            if self.vicon_marker_swarm_interface.ready():
                print("Planner received vicon marker swarm estimation")
                states = {"base": (self.robot_interface.q[:3], self.robot_interface.v[:3])}
                self.sot.update_planner(self.vicon_marker_swarm_interface.position, states)
            else:
                use_vicon_marker_swarm_data = False

                print("Planner did not receive vicon marker swarm estimation. Using config file to initialize SoT.")

        print("-----Checking Joy stick messages----- ")
        if self.use_joy:
            if self.joystick_interface.ready():
                print("Received joystick msg. Using joystick data.")
            else:
                self.use_joy = False
                print("Did not receive joystick msg.")


        if use_vicon_tool_data:
            self.planner_coord_transform(self.robot_interface.q, self.vicon_tool_interface.position, self.sot.planners)
        else:
            ee_pos, _ = self.robot.getEE(self.robot_interface.q)
            self.planner_coord_transform(self.robot_interface.q, ee_pos, self.sot.planners)

        rospy.Timer(rospy.Duration(0, int(1e8)), self._publish_planner_data)

        if self.use_joy:
            print("----- Press Square(Ps4) to start -----")
            while not self.start_end_button_interface.button == 1:
                rate.sleep()

            self.start_end_button_interface.reset_button()
        else:
            # print("Continue")
            input("----- Press Enter to start -----")

        # rospy.sleep(5.)
        self.sot.activatePlanners()
        t = rospy.Time.now().to_sec()
        t0 = t
        self.sot.started = True

        sot_num_plans = self.ctrl_config.get("task_num", None)
        if sot_num_plans is None:
            sot_num_plans = 1 if self.ctrl_config["type"][:2] == "ST" else 2
        while not self.ctrl_c:
            t = rospy.Time.now().to_sec()
            if self.ctrl_config["sdf_collision_avoidance_enabled"]:
                tm0 = time.perf_counter()
                status, map = self.map_interface.get_map()
                tm1 = time.perf_counter()

                t_get_map = tm1 - tm0
                
                if status:
                    map_latest = map
                else:
                    map_latest = None

            # open-loop command
            robot_states = (self.robot_interface.q, self.robot_interface.v)
            self.sot_lock.acquire()
            planners = self.sot.getPlanners(num_planners=sot_num_plans)
            self.sot_lock.release()
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

            for planner in planners:
                planner.updateRobotStates(robot_states)
                
            tc1 = time.perf_counter()
            u, acc, u_bar, v_bar = self.controller.control(t-t0, robot_states, planners, map_latest)
            tc2 = time.perf_counter()
            self.controller_log.log(20, "Controller Run Time: {}".format(tc2 - tc1))

            mpc_plan = v_bar
            N = mpc_plan.shape[0]
            t_mpc = np.arange(N) * self.mpc_dt
            mpc_plan_interp = interp1d(t_mpc, mpc_plan, axis=0, 
                                            bounds_error=False, fill_value="extrapolate")
            self.lock.acquire()
            self.mpc_plan = mpc_plan
            self.mpc_plan_time_stamp = t
            self.mpc_plan_interp = mpc_plan_interp
            self.lock.release()

            # publish data
            self._publish_mpc_data(self.controller)
            self._publish_trajectory_tracking_pt(t-t0, robot_states, planners)
            #if (ref_p is not None) and (ref_v is not None):
            #    self._publish_controller_reference(ref_p, ref_v)

            # Update Task Manager
            if use_vicon_tool_data:
                ee_states = (self.vicon_tool_interface.position, self.vicon_tool_interface.orientation)
            else:
                ee_states = self.robot.getEE(robot_states[0])
            states = {"base": (robot_states[0][:3], robot_states[1][:3]), "EE": ee_states}
            if self.use_joy:
                self.joystick_interface.button_lock.acquire()
                button = self.joystick_interface.button
                self.joystick_interface.button_lock.release()
                states["joy"] = button

            self.sot_lock.acquire()
            if self.sot.__class__.__name__ == "SoTSequentialTasks" and use_vicon_marker_swarm_data:
                # if t - t0 > 0 and t - t0 < 0.2:
                #     human_pos = np.array([[-3, -3, 0.8],
                #                  [-3, 3, 0.8],
                #                  [3, -0, 1.0]])
                #     self.sot.update_planner(human_pos, states)
                self.sot.update_planner(self.vicon_marker_swarm_interface.position, states)

            updated, _ = self.sot.update(t-t0, states)
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
            for planner in planners:
                if planner.type == "EE":
                    r_ew_wd, v_ew_wd  = planner.getTrackingPoint(t-t0, robot_states)
                elif planner.type == "base":
                    r_bw_wd, v_bw_wd = planner.getTrackingPoint(t-t0, robot_states)

                    if planner.name == "PartialPlanner":
                        ref_q_dot, ref_u = planner.getRefVelandAcc(t-t0)
                        self.logger.append("ref_vels", ref_q_dot)
                        self.logger.append("ref_accs", ref_u)
            if r_ew_wd is not None:
                self.logger.append("r_ew_w_ds", r_ew_wd)
            if v_ew_wd is not None:
                self.logger.append("v_ew_w_ds", v_ew_wd)
            if r_bw_wd is not None:
                if r_bw_wd.shape[0] == 2:
                    self.logger.append("r_bw_w_ds", r_bw_wd)

                elif r_bw_wd.shape[0] == 3:
                    self.logger.append("r_bw_w_ds", r_bw_wd[:2])
                    self.logger.append("yaw_bw_w_ds", r_bw_wd[2])
            if v_bw_wd is not None:
                if v_bw_wd.shape[0] == 2:
                    self.logger.append("v_bw_w_ds", v_bw_wd)
                elif  v_bw_wd.shape[0] == 3:
                    self.logger.append("v_bw_w_ds", v_bw_wd[:2])
                    self.logger.append("ω_bw_w_ds", v_bw_wd[2])

            self.logger.append("cmd_vels", u)
            self.logger.append("cmd_accs", acc)
            if self.ctrl_config["sdf_collision_avoidance_enabled"]:
                self.logger.append("time_get_map", t_get_map)

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
        log = self.controller.log
        for (key, val) in controller.log.items():
            logger.append("_".join(["mpc", key])+"s", val)

    def planner_coord_transform(self, q, ree, planners):
        R_wb = rotz(q[2])
        for planner in planners:
            P = np.zeros(3)
            if planner.frame_id == "base":
                P = np.hstack((q[:2],0))
            elif planner.frame_id == "EE":
                P = ree

            if planner.__class__.__name__ == "EESimplePlanner":
                planner.target_pos = R_wb @ planner.target_pos + P
                print(planner.target_pos)
            elif planner.__class__.__name__ == "EEPosTrajectoryCircle":
                planner.c = R_wb @ planner.c + P
                planner.plan['p'] = planner.plan['p'] @ R_wb.T + P
            elif planner.__class__.__name__ == "EEPosTrajectoryLine":
                planner.plan['p'] = planner.plan['p'] @ R_wb.T + P

            elif planner.__class__.__name__ == "BaseSingleWaypoint":
                planner.target_pos = (R_wb @ np.hstack((planner.target_pos, 0)))[:2] + P[:2]
            elif planner.__class__.__name__ == "BasePosTrajectoryCircle":
                planner.c = R_wb[:2,:2] @ planner.c + P[:2]
                planner.plan['p'] = planner.plan['p'] @ R_wb[:2, :2].T + P[:2]
            elif planner.__class__.__name__ == "BasePosTrajectoryLine" or planner.__class__.__name__ == "BasePosTrajectorySqaureWave":
                planner.plan['p'] = planner.plan['p'] @ R_wb[:2, :2].T + P[:2]

if __name__ == "__main__":
    rospy.init_node("controller_ros")

    node = ControllerROSNode()
    node.run()

