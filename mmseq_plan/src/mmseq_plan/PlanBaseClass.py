#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import logging
import casadi as ca
import numpy as np

from mmseq_utils.plot_casadi_time_optimal import decompose_X
from mmseq_utils.trajectory_generation import interpolate
from mmseq_utils.point_mass_computation_scripts.casadi_initial_guess import initial_guess_simple, initial_guess, slow_down_guess, slow_down_guess_only_ee
from mmseq_utils.plot_casadi_time_optimal import plot_obstacle_avoidance, compare_trajectories_casadi_plot, plot_motion_model

class Planner(ABC):
    def __init__(self, name, type, ref_type, ref_data_type, frame_id):
        self.py_logger = logging.getLogger("Planner")
        self.name = name
        # The following variables are for automatically 
        # (1) publishing rviz visualization data
        # (2) assigning the correct mpc cost function
        self.type = type                        # base or EE
        self.ref_type = ref_type                # waypoint vs trajectory
        self.ref_data_type = ref_data_type      # Vec2 vs Vec3
        self.frame_id = frame_id                # base or EE


    @abstractmethod
    def getTrackingPoint(self, t, robot_states=None):
        """ get tracking point for controllers

        :param t: time (s)
        :type t: float
        :param robot_states: (joint angle, joint velocity), defaults to None
        :type robot_states: tuple, optional
        
        :return: position, velocity
        :rtype: numpy array, numpy array
        """
        p = None
        v = None
        return p,v
    
    @abstractmethod
    def checkFinished(self, t, P):
        """check if the planner is finished 

        :param t: time since the controller started
        :type t: float
        :param P: EE position for EE planner, base position for base planner
        :type P: numpy array
        :return: true if the planner has finished, false otherwise
        :rtype: boolean
        """
        finished = True
        return finished

class TrajectoryPlanner(Planner):

    def _interpolate(self, t, plan):
        p,v = interpolate(t, plan)

        return p, v

class WholeBodyPlanner(Planner):
    def __init__(self):
        pass
    
    @abstractmethod
    def generatePlanFromConfig(self):
        pass

    @abstractmethod
    def initiliazePartialPlanners(self):
        pass
    
    def control(self, t):
        ''' Given a time, return an interpolated velocity'''
        q, q_dot, _ = self.interpolate(t)
        return q_dot
    
    def getRefVelandAcc(self, t):
        ''' Given a time, return an interpolated velocity and acceleration'''
        _, q_dot, q_dot_dot = self.interpolate(t)
        return q_dot, q_dot_dot

    
    def getTrackingPoint(self, t):
        ''' Given a time, return the end effector position and base position'''
        q, q_dot, _ = self.interpolate(t)

        end_effector_p = self.motion_class.end_effector_pose(q)
        end_effector_v = self.motion_class.compute_jacobian_whole(q) @ q_dot

        base_p = self.motion_class.base_xyz(q)
        base_v = self.motion_class.base_jacobian(q) @ q_dot
        return (end_effector_p, end_effector_v), (base_p, base_v)
    
    def checkFinished(self, t, ee_curr_pos):
        return t > self.tf

    def getBasePlanner(self):
        return self.base_planner
    
    def getEEPlanner(self):
        return self.ee_planner

    def getPlanners(self):
        return self.planners

    def slowDownTrajectory(self, start_state, end_ee, motion_model, forward_kinematic, N=50, obstacle_avoidance=None, self_avoidance=None):
        state_dim = self.motion_class.DoF

        X = ca.MX.sym(f'X', state_dim*3, N)
        t = ca.MX.sym(f't', 1)
        dt = t/N

        total_elements = state_dim*3
        vel_start_index = state_dim
        u_start_index = 2*state_dim

        g = []
        lbg = []
        ubg = []
        lbx = []
        ubx = []
        obj = t
        print(start_state)
        for i in range(N-1):
            cur_lbx = ca.DM.ones(total_elements)*-np.inf
            cur_ubx = ca.DM.ones(total_elements)*np.inf
            # constraints on position
            cur_lbx[:vel_start_index] = ca.horzcat(*self.q_min)
            cur_ubx[:vel_start_index] = ca.horzcat(*self.q_max)
            # constraints on velocity
            cur_lbx[vel_start_index:u_start_index] = ca.horzcat(*self.q_dot_min)
            cur_ubx[vel_start_index:u_start_index] = ca.horzcat(*self.q_dot_max)
            # constraints on acceleration
            cur_lbx[u_start_index:] = ca.horzcat(*self.u_min)
            cur_ubx[u_start_index:] = ca.horzcat(*self.u_max)
            # Jerk constraints
            # g.append((X[u_start_index:, i+1] - X[u_start_index:, i])/dt)
            # lbg.append(ca.vertcat(*self.jerk_min))
            # ubg.append(ca.vertcat(*self.jerk_max))
            if i==0:
                # g.append(X[:u_start_index, i] - start_state[:u_start_index])
                # lbg.append(ca.DM.zeros(2*state_dim))
                # ubg.append(ca.DM.zeros(2*state_dim))
                # cur_lbx[:u_start_index] = start_state[:u_start_index]
                # cur_ubx[:u_start_index] = start_state[:u_start_index]
                g.append(X[:u_start_index, i] -  start_state[:u_start_index] - dt*motion_model(start_state[:u_start_index], start_state[u_start_index:]))
                lbg.append(ca.DM.zeros(2*state_dim))
                ubg.append(ca.DM.zeros(2*state_dim))

            # Motion model
            g.append(X[:u_start_index, i+1] -  X[:u_start_index, i] - dt*motion_model(X[:u_start_index, i], X[u_start_index:, i]))
            lbg.append(ca.DM.zeros(2*state_dim))
            ubg.append(ca.DM.zeros(2*state_dim))
            lbx.append(cur_lbx)
            ubx.append(cur_ubx)
            # Obstacle constraints
            # if obstacle_avoidance is not None:
            #     obs = obstacle_avoidance(X[:vel_start_index, i]) - self.ground_collision_safety_margin
            #     g.append(obs)
            #     lbg.append(ca.DM.zeros(obs.shape[0]))
            #     ubg.append(ca.DM.ones(obs.shape[0])*np.inf)
            # if self_avoidance is not None:
            #     obs = self_avoidance(X[:vel_start_index, i]) - self.self_collision_safety_margin
            #     g.append(obs)
            #     lbg.append(ca.DM.zeros(obs.shape[0]))
            #     # lbg.append(ca.DM.ones(obs.shape[0])*-np.inf)
            #     ubg.append(ca.DM.ones(obs.shape[0])*np.inf)

            # As objective, minimize 
            # obj += ca.sumsqr(X[vel_start_index:u_start_index, i]) #+ ca.sumsqr(X[:vel_start_index, i] - self.starting_configuration)

        # Final point constraints
        cur_lbx = ca.DM.ones(total_elements)*-np.inf
        cur_ubx = ca.DM.ones(total_elements)*np.inf
        # constraints on position
        cur_lbx[:vel_start_index] = ca.horzcat(*self.q_min)
        cur_ubx[:vel_start_index] = ca.horzcat(*self.q_max)
        # constraints on velocity
        cur_lbx[vel_start_index:u_start_index] = ca.horzcat(*self.q_dot_min)
        cur_ubx[vel_start_index:u_start_index] = ca.horzcat(*self.q_dot_max)
        # constraints on acceleration
        cur_lbx[u_start_index:] = ca.horzcat(*self.u_min)
        cur_ubx[u_start_index:] = ca.horzcat(*self.u_max)

        cur_lbx[vel_start_index:u_start_index] = ca.DM.zeros(state_dim)
        cur_ubx[vel_start_index:u_start_index] = ca.DM.zeros(state_dim)
        # g.append(X[:vel_start_index, N-1] - self.starting_configuration)
        # lbg.append(ca.DM.zeros(state_dim))
        # ubg.append(ca.DM.ones(state_dim)*0)
        g.append(forward_kinematic(X[:vel_start_index, N-1]) - end_ee)
        lbg.append(ca.DM.zeros(len(end_ee)))
        ubg.append(ca.DM.zeros(len(end_ee)))
        # Obstacle constraints
        if self_avoidance is not None:
            self_avoid = self_avoidance(X[:vel_start_index, N-1]) - self.self_collision_safety_margin
            g.append(self_avoid)
            lbg.append(ca.DM.zeros(self_avoid.shape[0]))
            ubg.append(ca.DM.ones(self_avoid.shape[0])*np.inf)
        if obstacle_avoidance is not None:
            obs = obstacle_avoidance(X[:vel_start_index, N-1]) - self.ground_collision_safety_margin
            g.append(obs)
            lbg.append(ca.DM.zeros(obs.shape[0]))
            ubg.append(ca.DM.ones(obs.shape[0])*np.inf)
        # cur_lbx[:vel_start_index] = ca.horzcat(*self.starting_configuration)
        # cur_ubx[:vel_start_index] = ca.horzcat(*self.starting_configuration)
        lbx.append(cur_lbx)
        ubx.append(cur_ubx)
        X = X.reshape((-1, 1))

        OPT_variables = ca.vertcat(t, X)
        # create a zero initial guess
        # X0 = ca.DM.zeros((total_elements*N)+1)
        start_config = start_state[:state_dim].full().flatten()
        points_full = [self.motion_class.end_effector_pose(start_config).full().flatten()]
        points_full.append(end_ee)
        # X0_array, ts, N = initial_guess_simple(self.motion_class, points_full, start_config, 1, N, is_sequential_guess=True)
        # X0_array, ts, _ = initial_guess(self.motion_class, points_full, start_config, 1, N, is_sequential_guess=True)
        end_state = ca.vertcat(*self.starting_configuration, ca.DM.zeros(state_dim))
        # X0_array, tf = slow_down_guess(start_state, end_state, N, self.motion_class, a_bounds=(self.u_min, self.u_max))
        X0_array, tf = slow_down_guess_only_ee(start_state, end_ee, N, self.motion_class, a_bounds=(self.u_min, self.u_max))
        X0 = ca.vertcat(tf, *X0_array)

        opts = {'print_time': True, 'ipopt.print_level': 4}
        # opts = {'print_time': False, 'ipopt.sb': 'yes', 'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6, 'ipopt.print_level': 0, 'ipopt.max_iter': 10000}

        nlp = {'x': OPT_variables, 'f': obj, 'g': ca.vertcat(*g)}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        res = solver(x0=X0, lbx=ca.vertcat(0, *lbx), ubx=ca.vertcat(np.inf,*ubx), lbg=ca.vertcat(*lbg), ubg=ca.vertcat(*ubg))
        X_final = res['x']

        self.X_slow = X_final
        self.total_elements_slow = total_elements
        
        if solver.stats()['return_status'] != 'Solve_Succeeded':
            print("slow down failed",solver.stats()['return_status'])

        compare_trajectories_casadi_plot([X_final, X0], points_full, None, None, forward_kinematic, [total_elements, total_elements], [X_final[0], X0[0]], [N, N], q_size=state_dim, show=True, a_bounds=[(self.u_min, self.u_max), (self.u_min, self.u_max)], v_bounds=(self.q_dot_min, self.q_dot_max))
        plot_obstacle_avoidance([X_final, X0], self_avoidance, [total_elements, total_elements], q_size=state_dim, show=True, labels=["Slow Down", "Initial guess"], limit=self.self_collision_safety_margin)
        
        plot_motion_model([X_final, X0], motion_model, [total_elements, total_elements], q_size=state_dim, show=True, labels=["Slow Down", "Initial guess"], limit=0)


    def processResults(self):
        self.dts = []
        self.tfs = self.ts.full().flatten()
        for i in range(len(self.Ns)):
            self.dts.append(float(self.tfs[i]/self.Ns[i]))
        # print(self.Ns)
        qs, qs_dots, us = decompose_X(self.X, self.qs_num, self.total_elements)
        qs_zero, qs_dot_zero, us_zero = decompose_X(self.X0, self.qs_num, self.total_elements)
        self.qs, self.qs_dots, self.us = qs.T, qs_dots.T, us.T
        self.qs_zero, self.qs_dot_zero, self.us_zero = qs_zero.T, qs_dot_zero.T, us_zero.T
        if self.slowdown_enabled and self.X_slow is not None:
            print("Slowdown enabled")
            total_elements_slow = 3*self.qs_num
            qs_slow, qs_dots_slow, us_slow = decompose_X(self.X_slow, self.qs_num, total_elements_slow)
            self.qs_slow, self.qs_dots_slow, self.us_slow = qs_slow.T, qs_dots_slow.T, us_slow.T
            self.tfs = np.concatenate((self.tfs, np.array([float(self.X_slow[0])])), axis=0)
            self.dts.append(float(self.X_slow[0]/self.N))
            self.Ns.append(len(self.qs_slow))
            #exctend qs arrays
            self.qs = np.concatenate((self.qs, self.qs_slow), axis=0)
            self.qs_dots = np.concatenate((self.qs_dots, self.qs_dots_slow), axis=0)
            self.us = np.concatenate((self.us, self.us_slow), axis=0)
        self.tf = np.sum(self.tfs)

    def interpolate(self, t):
        offset=0
        if t > sum(self.tfs):
            return self.qs[-1], [0]*self.qs_num, [0]*self.qs_num
        for i in range(len(self.tfs)):
            if t < self.tfs[i]:
                break
            t -= self.tfs[i]
            offset += self.Ns[i]
        # Interpolate
        lower_index = int(t/self.dts[i]) + offset
        lower_index_time = int(t/self.dts[i]) 
        upper_index = lower_index_time + 1
        if lower_index_time >= (self.Ns[i]):
            return self.qs[self.Ns[i]-1], self.qs_dots[self.Ns[i]-1], self.us[self.Ns[i]-1]
        if upper_index >= (self.Ns[i]-1):
            return self.qs[lower_index], self.qs_dots[lower_index], self.us[lower_index]
        lower_time = lower_index_time*self.dts[i]
        upper_time = (lower_index_time +1)*self.dts[i]
        upper_index = upper_index + offset
        
        q = self.qs[lower_index] + (self.qs[upper_index] - self.qs[lower_index])*(t - lower_time)/(upper_time - lower_time)
        q_dot = self.qs_dots[lower_index] + (self.qs_dots[upper_index] - self.qs_dots[lower_index])*(t - lower_time)/(upper_time - lower_time)
        q_dot_dot = self.us[lower_index] + (self.us[upper_index] - self.us[lower_index])*(t - lower_time)/(upper_time - lower_time)
        return q, q_dot, q_dot_dot

class CasadiPartialPlanner(Planner):
    def __init__(self, qs, qs_dots, us, tfs, Ns, config, pose_calculator_func, jacobian_calculator_func):
        self.name = config["name"]
        self.type = config["type"]
        self.ref_type = "trajectory"
        self.ref_data_type = config["ref_data_type"]
        self.tracking_err_tol = config["tracking_err_tol"]
        self.frame_id = config["frame_id"]

        self.pose_calculator_func = pose_calculator_func
        self.jacobian_calculator_func = jacobian_calculator_func

        self.qs = qs
        self.qs_dots = qs_dots
        self.us = us
        self.tfs = tfs
        self.Ns = Ns
        self.dts = []
        self.tf = sum(self.tfs)
        for i in range(len(self.Ns)):
            self.dts.append(float(self.tfs[i]/self.Ns[i]))

        # generate self.plan["p"]
        self.plan = {"p": []}
        for i in range(len(self.qs)):
            self.plan["p"].append(self.pose_calculator_func(self.qs[i]).full().flatten())

    def interpolate(self, t):
        offset=0
        if t > self.tf:
            return self.qs[-1], [0]*len(self.qs[0]), [0]*len(self.qs[0])
        for i in range(len(self.tfs)):
            if t < self.tfs[i]:
                break
            t -= self.tfs[i]
            offset += self.Ns[i]
        # Interpolate
        lower_index = int(t/self.dts[i]) + offset
        lower_index_time = int(t/self.dts[i]) 
        upper_index = lower_index_time + 1
        if lower_index_time == (self.Ns[i]):
            return self.qs[self.Ns[i]-1], self.qs_dots[self.Ns[i]-1], self.us[self.Ns[i]-1]
        if upper_index >= (self.Ns[i]-1):
            return self.qs[lower_index], self.qs_dots[lower_index], self.us[lower_index]
        lower_time = lower_index_time*self.dts[i]
        upper_time = (lower_index_time +1)*self.dts[i]
        upper_index = upper_index + offset
        
        q = self.qs[lower_index] + (self.qs[upper_index] - self.qs[lower_index])*(t - lower_time)/(upper_time - lower_time)
        q_dot = self.qs_dots[lower_index] + (self.qs_dots[upper_index] - self.qs_dots[lower_index])*(t - lower_time)/(upper_time - lower_time)
        q_dot_dot = self.us[lower_index] + (self.us[upper_index] - self.us[lower_index])*(t - lower_time)/(upper_time - lower_time)
        return q, q_dot, q_dot_dot

    def getTrackingPoint(self, t, robot_states=None):
        ''' Given a time, return the end effector position and base position'''
        q, q_dot, _ = self.interpolate(t)

        p = self.pose_calculator_func(q)
        v = self.jacobian_calculator_func(q) @ q_dot

        return p.full().flatten(), v.full().flatten()

    def getRefVelandAcc(self, t):
        ''' Given a time, return an interpolated velocity and acceleration'''
        _, q_dot, q_dot_dot = self.interpolate(t)
        return q_dot, q_dot_dot
    
    def checkFinished(self, t, ee_curr_pos):
        return t > self.tf

    

