#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import numpy as np
import threading
import rospy
from nav_msgs.msg import Path
from mmseq_plan.PlanBaseClass import Planner, TrajectoryPlanner
from mmseq_utils.parsing import parse_number
from mmseq_utils.math import wrap_pi_scalar, wrap_pi_array
from mmseq_utils.trajectory_generation import sqaure_wave

class BaseSingleWaypoint(Planner):

    def __init__(self, planner_params):
        super().__init__(name=planner_params["name"],
                         type="base",
                         ref_type="waypoint",
                         ref_data_type="Vec2",
                         frame_id=planner_params["frame_id"])

        self.target_pos = np.array(planner_params["target_pos"])
        self.tracking_err_tol = np.array(planner_params["tracking_err_tol"])

        self.finished = False



    def getTrackingPoint(self, t, robot_states=None):
        return self.target_pos, np.zeros(2)

    def checkFinished(self, t, states):
        base_curr_pos = states[0][:2]
        err = np.linalg.norm(self.target_pos - base_curr_pos)
        if err < self.tracking_err_tol and not self.finished:
            self.py_logger.info(self.name + " planner finished.")
            self.finished = True

        return self.finished

    def reset(self):
        self.finished = False
        self.py_logger.info(self.name + " planner reset.")


    @staticmethod
    def getDefaultParams():
        config = {}
        config["name"] = "Base Position"
        config["planner_type"] = "BaseSingleWaypoint"
        config["frame_id"] = "base"
        config["target_pos"] = [0, 0]
        config["tracking_err_tol"] = 0.02

        return config
    
class BaseSingleWaypointPose(Planner):

    def __init__(self, planner_params):
        super().__init__(name=planner_params["name"],
                         type="base",
                         ref_type="waypoint",
                         ref_data_type="SE2",
                         frame_id=planner_params["frame_id"])

        self.target_pose = np.array(planner_params["target_pose"])
        self.tracking_err_tol = np.array(planner_params["tracking_err_tol"])

        self.finished = False



    def getTrackingPoint(self, t, robot_states=None):
        return self.target_pose, np.zeros(3)

    def checkFinished(self, t, states):
        base_curr_pos = states[0]
        err = np.linalg.norm(self.target_pose - base_curr_pos)
        if err < self.tracking_err_tol and not self.finished:
            self.py_logger.info(self.name + " planner finished.")
            self.finished = True

        return self.finished

    def reset(self):
        self.finished = False
        self.py_logger.info(self.name + " planner reset.")


    @staticmethod
    def getDefaultParams():
        config = {}
        config["name"] = "Base Pose"
        config["planner_type"] = "BaseSingleWaypointPose"
        config["frame_id"] = "base"
        config["target_pos"] = [0, 0, 0]
        config["tracking_err_tol"] = 0.02

        return config

class BasePosTrajectoryCircle(TrajectoryPlanner):
    def __init__(self, config):
        super().__init__(name=config["name"],
                        type="base",
                        ref_type="trajectory",
                        ref_data_type="Vec2",
                        frame_id=config["frame_id"])

        self.tracking_err_tol = config["tracking_err_tol"]

        self.finished = False
        self.started = False
        self.start_time = 0

        self.T = config["period"]
        self.omega = np.pi * 2 / self.T
        self.c = np.array(config["center"])
        self.r = config["radius"]
        self.phi = parse_number(config["angular_offset"])
        self.round = int(config["round"])

        self.dt = 0.01
        self.N = int(self.T * self.round / self.dt)
        self.plan = self._generatePlan()

    def _generatePlan(self):
        plan_pos = np.repeat([self.c], self.N, axis=0)

        ts = np.linspace(0, self.T * self.round, self.N)
        ptx = self.r * np.cos(self.omega * ts + self.phi)
        pty = self.r * np.sin(self.omega * ts + self.phi)

        plan_pos[:, 0] += ptx
        plan_pos[:, 1] += pty

        plan_vel = np.zeros_like(plan_pos)
        vx = -self.r * np.sin(self.omega * ts + self.phi) * self.omega
        vy = self.r * np.cos(self.omega * ts + self.phi) * self.omega
        plan_vel[:, 0] = vx
        plan_vel[:, 1] = vy

        return {"t": ts, "p": plan_pos, "v": plan_vel}

    def getTrackingPoint(self, t, robot_states=None):

        if self.started and self.start_time == 0:
            self.start_time = t

        te = t - self.start_time

        p, v = self._interpolate(te, self.plan)

        return p, v

    def checkFinished(self, t, states):
        base_curr_pos = states[0][:2]

        if t - self.start_time > self.T * (self.round - 1):
            if np.linalg.norm(base_curr_pos - self.plan[0]) < self.tracking_err_tol:
                self.finished = True
                self.py_logger.info(self.name + " Planner Finished")
        return self.finished

    def reset(self):
        self.finished = False
        self.started = False
        self.py_logger.info(self.name + " planner reset.")

class BasePosTrajectoryLine(TrajectoryPlanner):
    def __init__(self, config):
        super().__init__(name=config["name"],
                        type="base",
                        ref_type="trajectory",
                        ref_data_type="Vec2",
                        frame_id=config["frame_id"])

        self.tracking_err_tol = config["tracking_err_tol"]
        self.end_stop = config.get("end_stop", False)

        self.finished = False
        self.started = False
        self.start_time = 0

        self.initial_pos = np.array(config["initial_pos"])
        self.target_pos = np.array(config["target_pos"])
        self.cruise_speed = config["cruise_speed"]

        self.dt = 0.01
        self.plan = self._generatePlan()


    def regeneratePlan(self):
        self.plan = self._generatePlan()
        self.start_time = 0

    def _generatePlan(self):
        self.T = np.linalg.norm(self.initial_pos - self.target_pos) / self.cruise_speed

        ts = np.linspace(0, self.T, int(self.T/self.dt)).reshape((-1, 1))
        n = (self.target_pos - self.initial_pos) / np.linalg.norm(self.initial_pos - self.target_pos)
        plan_pos = n * ts * self.cruise_speed + self.initial_pos

        plan_vel = np.tile(n * self.cruise_speed, (int(self.T/self.dt), 1))

        return {'t': ts, 'p': plan_pos, 'v': plan_vel}

    def getTrackingPoint(self, t, robot_states=None):

        if self.started and self.start_time == 0:
            self.start_time = t

        te = t - self.start_time

        p, v = self._interpolate(te, self.plan)

        return p, v

    def checkFinished(self, t, states):
        base_curr_pos = states[0][:2]
        base_curr_vel = states[1][:2]
        pos_cond = np.linalg.norm(base_curr_pos - self.plan['p'][-1]) < self.tracking_err_tol
        vel_cond = np.linalg.norm(base_curr_vel) < 1e-2
        if (not self.end_stop and pos_cond) or (self.end_stop and pos_cond and vel_cond):
            self.finished = True
            self.py_logger.info(self.name + " Planner Finished Position error {}".format(np.linalg.norm(base_curr_pos - self.plan['p'][-1])))
        return self.finished

    def reset(self):
        self.finished = False
        self.start_time = 0
        self.py_logger.info(self.name + " planner reset.")

    @staticmethod
    def getDefaultParams():
        config = {}
        config["name"] = "Base Position"
        config["planner_type"] = "BasePosTrajectoryLine"
        config["frame_id"] = "base"
        config["initial_pos"] = [0, 0]
        config["target_pos"] = [0, 0]
        config["cruise_speed"] = 0.5
        config["tracking_err_tol"] = 0.02

        return config

class BasePoseTrajectoryLine(TrajectoryPlanner):
    def __init__(self, config):
        super().__init__(name=config["name"],
                        type="base",
                        ref_type="trajectory",
                        ref_data_type="Vec3",
                        frame_id=config["frame_id"])
        self.tracking_err_tol = config["tracking_err_tol"]
        self.end_stop = config.get("end_stop", False)

        self.finished = False
        self.started = False
        self.start_time = 0

        self.initial_pose = np.array(config["initial_pose"])
        self.target_pose = np.array(config["target_pose"])
        self.cruise_speed = config["cruise_speed"]
        self.yaw_speed = config["yaw_speed"]

        self.dt = 0.01
        self.plan = self._generatePlan()

    def regeneratePlan(self):
        self.plan = self._generatePlan()
        self.start_time = 0

    def _generatePlan(self):
        initial_pos = self.initial_pose[:2]
        target_pos = self.target_pose[:2]
        initial_heading = wrap_pi_scalar(self.initial_pose[2])
        target_heading = wrap_pi_scalar(self.target_pose[2])

        T_pos = np.linalg.norm(initial_pos - target_pos) / self.cruise_speed

        ts = np.linspace(0, T_pos, int(T_pos/self.dt)).reshape((-1, 1))
        n = (target_pos - initial_pos) / np.linalg.norm(initial_pos - target_pos)
        plan_pos = n * ts * self.cruise_speed + initial_pos

        plan_vel = np.tile(n * self.cruise_speed, (int(T_pos/self.dt), 1))

        heading_diff = target_heading - initial_heading
        print("target_heading: {}, initial_heading: {}".format(target_heading, initial_heading))
        # if heading_diff > np.pi:
        #     heading_diff -= 2 * np.pi
        # elif heading_diff < -np.pi:
        #     heading_diff += 2 * np.pi

        omega = np.sign(heading_diff) * self.yaw_speed
        if omega != 0:
            T_heading = heading_diff / omega
        else:
            T_heading = T_pos
        ts_heading = np.linspace(0, T_heading, int(T_heading/self.dt)).reshape((-1, 1))
        plan_heading = omega * ts_heading + initial_heading
        plan_omega = np.ones_like(plan_heading) * omega

        d = plan_pos.shape[0] - plan_heading.shape[0]
        if d > 0:
            padding = np.ones((d, 1)) * plan_heading[-1]
            plan_heading = np.vstack((plan_heading, padding))
            plan_omega = np.vstack((plan_omega, np.zeros((d, 1))))

            t = ts

        else:
            d = -d
            padding = np.ones((d, 2)) * plan_pos[-1]
            plan_pos = np.vstack((plan_pos, padding))
            plan_vel = np.vstack((plan_vel, np.zeros((d, 2))))

            t = ts_heading

        p = np.hstack((plan_pos, plan_heading))
        v = np.hstack((plan_vel, plan_omega))


        return {'t': t, 'p': p, 'v': v}

    def getTrackingPoint(self, t, robot_states=None):

        if self.started and self.start_time == 0:
            self.start_time = t

        te = t - self.start_time

        p, v = self._interpolate(te, self.plan)
        return p, v

    def checkFinished(self, t, states):
        base_curr_pose = states[0]
        base_curr_vel = states[1]
        pos_cond = np.linalg.norm(base_curr_pose - self.plan['p'][-1]) < self.tracking_err_tol
        vel_cond = np.linalg.norm(base_curr_vel) < 1e-2
        if (not self.end_stop and pos_cond) or (self.end_stop and pos_cond and vel_cond):
            self.finished = True
            self.py_logger.info(self.name + " Planner Finished Pose error {}".format(np.linalg.norm(base_curr_pose - self.plan['p'][-1])))
        return self.finished

    def reset(self):
        self.finished = False
        self.start_time = 0
        self.py_logger.info(self.name + " planner reset.")

    @staticmethod
    def getDefaultParams():
        config = {}
        config["name"] = "Base Pose"
        config["planner_type"] = "BasePoseTrajectoryLine"
        config["frame_id"] = "base"
        config["initial_pose"] = [0, 0, 0]
        config["target_pose"] = [0, 0, 0]
        config["cruise_speed"] = 0.5
        config["yaw_speed"] = 0.5
        config["tracking_err_tol"] = 0.02

        return config

class BasePosTrajectorySqaureWave(TrajectoryPlanner):
    def __init__(self, config):
        super().__init__(name=config["name"],
                        type="base",
                        ref_type="trajectory",
                        ref_data_type="Vec2",
                        frame_id=config["frame_id"])
        self.tracking_err_tol = config["tracking_err_tol"]

        self.finished = False
        self.started = False
        self.start_time = 0

        self.dt = 0.01
        self.plan = sqaure_wave(config["peak_pos"], config["valley_pos"], config["period"], config["round"], self.dt)

    def getTrackingPoint(self, t, robot_states=None):

        if self.started and self.start_time == 0:
            self.start_time = t

        te = t - self.start_time

        p, v = self._interpolate(te, self.plan)

        return p, v

    def checkFinished(self, t, states):
        base_curr_pos = states[0][:2]

        if np.linalg.norm(base_curr_pos - self.plan['p'][-1]) < self.tracking_err_tol:
            self.finished = True
            self.py_logger.info(self.name + " Planner Finished")
        return self.finished

    def reset(self):
        self.finished = False
        self.started = False
        self.start_time = 0

    @staticmethod
    def getDefaultParams():
        config = {}
        config["name"] = "Base Position"
        config["planner_type"] = "BasePosTrajectorySqaureWave"
        config["frame_id"] = "base"
        config["peak_pos"] = [0, 0]
        config["valley_pos"] = [0, 0]
        config["period"] = 10
        config["round"] = 1
        config["tracking_err_tol"] = 0.02

        return config
    
class ROSTrajectoryPlanner(TrajectoryPlanner):
    def __init__(self, config):
        super().__init__(name=config["name"],
                        type="base",
                        ref_type="path",
                        ref_data_type="SE2",
                        frame_id=config["frame_id"])
        
        print("ROSTrajectoryPlanner")
        print(config)
        self.tracking_err_tol = config["tracking_err_tol"]
        self.end_stop = config.get("end_stop", False)

        self.finished = False
        self.started = False
        self.start_time = -1
        self.plan_available = False

        self.cruise_speed = config["cruise_speed"]
        self.yaw_speed = config["yaw_speed"]
        self.yaw_accel = config["yaw_accel"]

        self.ref_traj_duration = config["ref_traj_duration"]
        self.dt = 0.01
        self.traj_length = int(self.ref_traj_duration / self.dt)
        self.plan = None
        self.lock = threading.Lock()

        self.path_sub = rospy.Subscriber("/planned_global_path", Path, self._path_callback, queue_size=1)

    def _generatePlan(self, start_time, raw_points, raw_headings):
        # given a set of path points and heading, generate a plan based on cruise speed
        # Compute cumulative distances along the path
        self.start_time = start_time

        # find the closest point on the path and truncate the path
        if self.robot_states is not None:
            base_curr_pos = self.robot_states[0][:2]
            dists_to_robot = np.linalg.norm(raw_points - base_curr_pos, axis=1)
            min_idx = np.argmin(dists_to_robot)
            #print('min dist to robot: ', np.min(dists_to_robot))
            if len(raw_points) - min_idx < 2:
                min_idx = max(0, len(raw_points) - 2)
                
            raw_points = raw_points[min_idx:, :]
            raw_headings = raw_headings[min_idx:]

        distances = np.linalg.norm(np.diff(raw_points, axis=0), axis=1)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

        # Total distance of the path
        total_distance = cumulative_distances[-1]

        # Resample the path based on constant velocity and dt
        time = np.arange(0, total_distance / self.cruise_speed, self.dt)
        if len(time) == 0:
            time = np.array([0])

        if len(time) < self.traj_length:
            # pad time by extrapolating with self.dt from last time
            time = np.append(time, np.arange(time[-1] + self.dt, self.ref_traj_duration, self.dt))
            # also pad raw_points and raw_headings with the last value
            raw_points = np.vstack((raw_points, np.tile(raw_points[-1], (self.traj_length - len(raw_points), 1))))
            raw_headings = np.append(raw_headings, np.tile(raw_headings[-1], (self.traj_length - len(raw_headings), 1)))
            # pad distances with 0
            distances = np.append(distances, np.zeros(self.traj_length - len(distances)))
            # pad cumulative_distances with the last value
            cumulative_distances = np.append(cumulative_distances, np.tile(cumulative_distances[-1], (self.traj_length - len(cumulative_distances))))

        time = time[:self.traj_length]
        points = raw_points[:self.traj_length, :]
        headings = raw_headings[:self.traj_length]

        new_x = np.interp(time * self.cruise_speed, cumulative_distances, points[:self.traj_length, 0]).reshape(-1, 1)
        new_y = np.interp(time * self.cruise_speed, cumulative_distances, points[:self.traj_length, 1]).reshape(-1, 1)

        #print('trajectory length: ', len(time))

        # Interpolate the headings based on the new time samples
        new_desired_headings = np.interp(time * self.cruise_speed, cumulative_distances, headings).reshape(-1, 1)
        velocities = np.zeros((self.traj_length, 3))

        if self.robot_states is not None: # smooth the heading transition
            # Calculate actual headings by capping yaw change per time step
            current_yaw = self.robot_states[0][2]
            current_yaw_speed = self.robot_states[1][2]
            for i in range(self.traj_length-1):
                yaw_diff = wrap_pi_scalar(new_desired_headings[i] - current_yaw)

                # Compute the desired yaw speed
                desired_yaw_speed = yaw_diff / self.dt
                
                # Limit the yaw speed to the maximum allowed yaw speed
                if abs(desired_yaw_speed) > self.yaw_speed:
                    desired_yaw_speed = np.sign(desired_yaw_speed) * self.yaw_speed
        

                # Compute the required yaw acceleration to reach the desired yaw speed
                yaw_acc = (desired_yaw_speed - current_yaw_speed) / self.dt
        

                # Limit the yaw acceleration to the maximum allowed angular acceleration
                if abs(yaw_acc) > self.yaw_accel:
                    yaw_acc = np.sign(yaw_acc) * self.yaw_accel

                current_yaw_speed += yaw_acc * self.dt

                # Limit the yaw speed again (in case acceleration limit was applied)
                if abs(current_yaw_speed) > self.yaw_speed:
                    current_yaw_speed = np.sign(current_yaw_speed) * self.yaw_speed

                current_yaw += current_yaw_speed * self.dt
                current_yaw = wrap_pi_scalar(current_yaw)  # Keep heading in [-pi, pi]
                new_desired_headings[i] = current_yaw

                velocities[i,0] = (new_x[i+1] - new_x[i]) / self.dt
                velocities[i,1] = (new_y[i+1] - new_y[i]) / self.dt
                velocities[i,2] = current_yaw_speed

        poses = np.hstack((new_x, new_y, new_desired_headings))

        self.close_to_finish = (poses[-1,:] == poses[-2,:]).all() and np.linalg.norm(self.robot_states[0][:2] - poses[-1,:2]) < 0.1 and np.abs(wrap_pi_scalar(self.robot_states[0][2] - poses[-1,2])) < 0.1

        return {'t': time,'s': time * self.cruise_speed, 'p': poses, 'v': velocities}

    def _path_callback(self, msg):
        time = msg.header.stamp.to_sec() # this is ros time in seconds
        path = np.array([[pose.pose.position.x, pose.pose.position.y] for pose in msg.poses]).reshape(-1, 2)
        heading = np.array([np.arctan2(2.0 * (pose.pose.orientation.w * pose.pose.orientation.z), 1.0 - 2.0 * (pose.pose.orientation.z * pose.pose.orientation.z)) for pose in msg.poses]) 

        plan = self._generatePlan(time, path, heading)

        self.lock.acquire()
        self.plan = copy.deepcopy(plan)
        self.lock.release()
        self.plan_available = True


    def getTrackingPoint(self, t, robot_states):
        # return self.getTrackingPointByStates(robot_states)
        #return self.getTrackingPointByTime(t)
        ps, vs = self.getTrackingPointArray(robot_states, 2, 0.1)
        return ps[0], vs[0]
    
    def getTrackingPointArray(self, robot_states, num_pts, dt):
        base_curr_pos = robot_states[0][:2]
        # search for the closest point on the path
        min_dist = np.inf
        min_idx = 0
        self.lock.acquire()
        plan = copy.deepcopy(self.plan)
        self.lock.release()

        for i in range(len(plan['p'])):
            dist = np.linalg.norm(base_curr_pos - plan['p'][i][:2])
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        s0 = plan['s'][min_idx]
        s = s0 + np.arange(num_pts) * dt * self.cruise_speed
        pos = [np.interp(s, plan['s'], plan['p'][:,i]) for i in range(3)]
        vel = [np.interp(s, plan['s'], plan['v'][:,i]) for i in range(3)]

        pos = np.array(pos).T
        vel = np.array(vel).T
        return pos, vel

    def getTrackingPointByTime(self, t):
        te = t - self.start_time
        p, v = self._interpolate(te, self.plan)

        return p, v
    
    def getTrackingPointByStates(self, states):
        base_curr_pos = states[0][:2]
        # search for the closest point on the path
        min_dist = np.inf
        min_idx = 0
        for i in range(len(self.plan['p'])):
            dist = np.linalg.norm(base_curr_pos - self.plan['p'][i][:2])
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        return self.plan['p'][min_idx], None

    def checkFinished(self, t, states):
        base_curr_pos = states[0]

        if np.linalg.norm(base_curr_pos - self.plan['p'][-1]) < self.tracking_err_tol:
            self.finished = True
        return self.finished

    def ready(self):
        return self.plan_available