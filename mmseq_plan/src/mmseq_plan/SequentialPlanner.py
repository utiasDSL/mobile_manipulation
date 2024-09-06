import casadi as ca
import numpy as np
import time
import datetime
import pickle
import os

from mmseq_utils.plot_casadi_time_optimal import decompose_X
from mmseq_utils.point_mass_computation_scripts.casadi_initial_guess import initial_guess, initial_guess_simple
import mobile_manipulation_central as mm

# for now import the mobile manipulators, in future we should be able to import a file and not having to update this if more motion classses are introduced
# todo this we could declare all the motion classes in a file and import that file ? 
import mmseq_control.mobile_manipulator_point_mass.mobile_manipulator_class as MobileManipulatorPointMass
import mmseq_control.robot as robot
from mmseq_plan.PlanBaseClass import WholeBodyPlanner, CasadiPartialPlanner
from mmseq_utils.parsing import load_config, parse_ros_path, parse_array
from pyb_utils.frame import debug_frame_world

class SequentialPlanner(WholeBodyPlanner):
    def __init__(self, config, motion_class=None, file_path=None):
        if motion_class is not None:
            self.motion_class = motion_class
            self.config = None
            self.file_path = file_path
        else:
            possibly_motion_class = getattr(MobileManipulatorPointMass, config["motion_class_type"], None)
            if possibly_motion_class is None:
                possibly_motion_class = getattr(robot, config["motion_class_type"], None)
            if possibly_motion_class is None:
                raise ValueError("Motion class {} not found".format(config["motion_class_type"]))
            # take path from yaml that provide package and location given package
            config_path = parse_ros_path(config["motion_class_config_file"])
            robot_config = load_config(config_path)["controller"]

            self.motion_class = possibly_motion_class(robot_config)
            self.points = config['target_points']
            self.prediction_horizon = config['prediction_horizon']
            self.N = config['N']
            self.dt_config = config['dt']
            self.init_config = config['initialization']
            self.obs_avoidance = config['obstacle_avoidance']
            self.self_collision_safety_margin = config["collision_safety_margin"]["self"]
            # print(self.starting_configuration)
            self.starting_configuration = mm.load_home_position(config.get("home", "default"))
            self.config = config
            self.file_path = config["file_path"]
            self.load_save_generate = config["load_save_generate"]
            self.file_name = config["file_name"]
            self.slowdown_enabled = config["slowdown_enabled"]
            # for point in self.points:
            #     debug_frame_world(0.5, point, line_width=3)

        self.X_slow = None
        self.qs_num = self.motion_class.DoF
        self.q_max = self.motion_class.ub_x[:self.qs_num]
        self.q_min = self.motion_class.lb_x[:self.qs_num]
        self.q_dot_max = self.motion_class.ub_x[self.qs_num:]
        self.q_dot_min = self.motion_class.lb_x[self.qs_num:]
        self.u_max = self.motion_class.ub_u
        self.u_min = self.motion_class.lb_u
        self.motion_model = self.motion_class.ssSymMdl["fmdl"]
        self.forward_kinematic = self.motion_class.end_effector_pose_func()

    @classmethod
    def initializeFromMotionClass(self, motion_class, file_path=os.path.join(os.getcwd(), "planner_solutions")):
        return SequentialPlanner(None, motion_class, file_path)


    def setup_single_problem(self, start, goal, motion_model, forward_kinematic, N=100, d_tol=0.01, initial_point=False, state_dim=2, iteration=0, obstacles_avoidance=None, obstacles=[], end_zero_velocity=False):
        '''Function that generates the constraints needed to slove an NLP to minimize the time taken for a robot to move between two points'''
        # Check dimension of the points
        X = ca.MX.sym(f'X_{iteration}', state_dim*3, N)
        t = ca.MX.sym(f't_{iteration}', 1)
        dt = t/N

        total_elements = state_dim*3
        vel_start_index = state_dim
        u_start_index = 2*state_dim

        g = []
        lbg = []
        ubg = []
        lbx = []
        ubx = []
        
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
            if i==0:
                if initial_point:
                    cur_lbx[:vel_start_index] = start[:vel_start_index]
                    cur_ubx[:vel_start_index] = start[:vel_start_index]
                    if start[vel_start_index] is not None:
                        cur_lbx[vel_start_index:u_start_index] = start[vel_start_index:]
                        cur_ubx[vel_start_index:u_start_index] = start[vel_start_index:]
                else:
                    # first point is the last of previous iteration so we need to set such identity
                    g.append(X[:u_start_index, i] -  start[:u_start_index] - dt*motion_model(start[:u_start_index], start[u_start_index:]))
                    lbg.append(ca.DM.zeros(2*state_dim))
                    ubg.append(ca.DM.zeros(2*state_dim))

            # Motion model
            g.append(X[:u_start_index, i+1] -  X[:u_start_index, i] - dt*motion_model(X[:u_start_index, i], X[u_start_index:, i]))
            lbg.append(ca.DM.zeros(2*state_dim))
            ubg.append(ca.DM.zeros(2*state_dim))
            lbx.append(cur_lbx)
            ubx.append(cur_ubx)
            # Obstacle constraints
            if obstacles_avoidance is not None:
                obs = obstacles_avoidance(X[:vel_start_index, i]) - self.self_collision_safety_margin
                g.append(obs)
                lbg.append(ca.DM.zeros(obs.shape[0]))
                # lbg.append(ca.DM.ones(obs.shape[0])*-np.inf)
                ubg.append(ca.DM.ones(obs.shape[0])*np.inf)

        # Final point constraints
        cur_lbx = ca.DM.ones(total_elements)*-np.inf
        cur_ubx = ca.DM.ones(total_elements)*np.inf
        g.append(forward_kinematic(X[:vel_start_index, N-1]) - goal)
        lbg.append(ca.DM.zeros(len(goal)))
        ubg.append(ca.DM.zeros(len(goal)))
        cur_lbx[:vel_start_index] = ca.horzcat(*self.q_min)
        cur_ubx[:vel_start_index] = ca.horzcat(*self.q_max)
        if not end_zero_velocity:
            cur_lbx[vel_start_index:u_start_index] = ca.horzcat(*self.q_dot_min)
            cur_ubx[vel_start_index:u_start_index] = ca.horzcat(*self.q_dot_max)
        else:
            cur_lbx[vel_start_index:u_start_index] = ca.DM.zeros(state_dim)
            cur_ubx[vel_start_index:u_start_index] = ca.DM.zeros(state_dim)
        cur_lbx[u_start_index:] = ca.horzcat(*self.u_min)
        cur_ubx[u_start_index:] = ca.horzcat(*self.u_max)
        lbx.append(cur_lbx)
        ubx.append(cur_ubx)
        X = X.reshape((-1, 1))
        return X, g, lbg, ubg, lbx, ubx, t      

    def optimize_sequential(self, points, prediction_horizon, X0, motion_model, forward_kinematic, Ns, d_tol=0.01, obstacles_avoidance=None):
        X = []
        g = []
        lbg = []
        ubg = []
        lbx = [0]*prediction_horizon #contains the time for each problem
        ubx = [np.inf]*prediction_horizon #contains the time for each problem
        ts = []
        state_dim = self.motion_class.DoF
        obj =0
        for i in range(prediction_horizon):
            is_initial_point = False
            if i == 0:
                is_initial_point = True
                start_point = X0[prediction_horizon:prediction_horizon+2*state_dim]
            else:
                start_point = X[-3*state_dim:]
            X_i, g_i, lbg_i, ubg_i, lbx_i, ubx_i, t_i = self.setup_single_problem(start_point, points[i+1], motion_model, forward_kinematic, N=Ns[i], d_tol=d_tol, initial_point=is_initial_point, state_dim=state_dim, iteration=i, obstacles_avoidance=obstacles_avoidance)
            X = ca.vertcat(X, X_i)
            g.extend(g_i)
            lbg.extend(lbg_i)
            ubg.extend(ubg_i)
            lbx.extend(lbx_i)
            ubx.extend(ubx_i)
            ts.append(t_i)
            obj += t_i
            
        OPT_variables = ca.vertcat(*ts, X)
        opts = {'print_time': False, 'ipopt.print_level': 0}
        opts = {'print_time': False, 'ipopt.sb': 'yes', 'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6, 'ipopt.print_level': 0, 'ipopt.max_iter': 10000}

        nlp = {'x': OPT_variables, 'f': obj, 'g': ca.vertcat(*g)}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        res = solver(x0=X0, lbx=ca.vertcat(*lbx), ubx=ca.vertcat(*ubx), lbg=ca.vertcat(*lbg), ubg=ca.vertcat(*ubg))
        X_dim = 3*state_dim
        result = res['x']
        final_time = ca.sum1(result[:prediction_horizon])
        X_final = ca.vertcat(final_time, result[prediction_horizon:])
        if solver.stats()['return_status'] != 'Solve_Succeeded':
            print(solver.stats()['return_status'])
        # print dts
        # for i in range(len(Ns)):
        #     print(result[i]/Ns[i])
        return X_final, X_dim, result[:prediction_horizon]
    
    def generateTrajectory(self, points, starting_configuration, prediction_horizon, N=100, t_bound=np.inf, obs_avoidance=True, init_config='Pontryagin', base_scaling_factor=0.1, ee_scaling_factor=0.5):
        self.u_min = self.motion_class.lb_u*base_scaling_factor
        self.u_max = self.motion_class.ub_u*base_scaling_factor
        self.u_min[3:] = self.motion_class.lb_u[3:]*ee_scaling_factor
        self.u_max[3:] = self.motion_class.ub_u[3:]*ee_scaling_factor
        points_full = [self.motion_class.end_effector_pose(starting_configuration).full().flatten()]
        points_full.extend(points)
        if init_config == 'Pontryagin':
            start_time = time.time()
            X0_array, ts, Ns = initial_guess(self.motion_class, points_full, starting_configuration, prediction_horizon, N, is_sequential_guess=True)
            self.initialization_time = time.time()-start_time
        elif init_config == 'InverseKinematics':
            start_time = time.time()
            X0_array, ts, Ns = initial_guess_simple(self.motion_class, points_full, starting_configuration, prediction_horizon, N, is_sequential_guess=True)
            self.initialization_time = time.time()-start_time
        else:
            raise ValueError(f"Invalid initialization method {init_config}")
        self.Ns = Ns
        X0 = ca.vertcat(*ts, *X0_array)
        obstacles_avoidance = None
        if obs_avoidance:
            obstacles_avoidance = self.motion_class.casadi_model_interface.signedDistanceSymMdlsPerGroup["self"]
        start_time = time.time()
        X, total_elements, ts = self.optimize_sequential(points_full, prediction_horizon, X0, self.motion_model, self.forward_kinematic, Ns, obstacles_avoidance=obstacles_avoidance)
        self.optimization_time = time.time()-start_time
        self.ts = ts
        self.X0 = ca.vertcat(float(ts[-1]), *X0_array)
        self.X = X
        self.total_elements = total_elements
        self.points = points
        self.prediction_horizon = prediction_horizon
        self.starting_configuration = starting_configuration
        self.init_config = init_config
        self.obs_avoidance = obs_avoidance
        return X, total_elements

    def generatePlanFromConfig(self):
        obstacles_avoidance = None
        if self.obs_avoidance:
            obstacles_avoidance = self.motion_class.casadi_model_interface.signedDistanceSymMdlsPerGroup["self"]
        if self.load_save_generate == "load":
            self.loadSolution(self.file_name)
            self.processResults()
            self.initiliazePartialPlanners()
        elif self.load_save_generate == "generate":
            X, total_elements = self.generateTrajectory(self.points, self.starting_configuration, self.prediction_horizon, N=self.N, t_bound=np.inf, init_config=self.init_config, obs_avoidance=self.obs_avoidance)
            if self.slowdown_enabled:
                self.slowDownTrajectory(X[-3*self.motion_class.DoF:], self.motion_class.end_effector_pose(self.starting_configuration).full().flatten(), self.motion_model, self.forward_kinematic, obstacle_avoidance=obstacles_avoidance)
            self.processResults()
            self.initiliazePartialPlanners()
        elif self.load_save_generate == "save":
            X, total_elements = self.generateTrajectory(self.points, self.starting_configuration, self.prediction_horizon, N=self.N, t_bound=np.inf, init_config=self.init_config, obs_avoidance=self.obs_avoidance)
            if self.slowdown_enabled:
                self.slowDownTrajectory(X[-3*self.motion_class.DoF:], self.motion_class.end_effector_pose(self.starting_configuration).full().flatten(), self.motion_model, self.forward_kinematic, obstacle_avoidance=obstacles_avoidance)
            self.saveSolution(name=self.file_name)

    def saveSolution(self, name=None):
        if name is None:
            # set to default name
            name = f'sequential_plan_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

        path = os.path.join(self.file_path, name)
        # check if directory exists
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)
        with open(path, 'wb') as f:
            pickle.dump((self.X, self.total_elements, self.ts, self.Ns, self.points, self.prediction_horizon, self.starting_configuration, self.X0, self.init_config, self.initialization_time, self.optimization_time, self.X_slow), f)

    def loadSolution(self, name):
        path = os.path.join(self.file_path, name)
        with open(path, 'rb') as f:
            self.X, self.total_elements, self.ts, self.Ns, self.points, self.prediction_horizon, self.starting_configuration, self.X0, self.init_config, self.initialization_time, self.optimization_time, self.X_slow = pickle.load(f)

    def returnWaypointConfigurations(self):
        offset = 0
        configurations = []
        initial_configurations = []
        for i in range(len(self.Ns)):
            index = offset + self.Ns[i] -1
            configurations.append(self.qs[index])
            initial_configurations.append(self.qs_zero[index])
            offset += self.Ns[i]

        return configurations, initial_configurations

    def initiliazePartialPlanners(self):
        if self.config is None:
            base_config = {"name": "PartialPlanner", "type": "base", "tracking_err_tol": 0.02, "frame_id": "base", "ref_data_type": "Vec2"}
            ee_config = {"name": "PartialPlanner", "type": "EE", "tracking_err_tol": 0.02, "frame_id": "EE", "ref_data_type": "Vec3"}

        else:
            base_config = {"name": "PartialPlanner", "type": "base", "tracking_err_tol": self.config["tracking_err_tol"], "frame_id": "base", "ref_data_type": "Vec2"}
            ee_config = {"name": "PartialPlanner", "type": "EE", "tracking_err_tol": self.config["tracking_err_tol"], "frame_id": "EE", "ref_data_type": "Vec3"}

        self.base_planner = CasadiPartialPlanner(self.qs, self.qs_dots, self.us, self.tfs, self.Ns, base_config, self.motion_class.base_xyz, self.motion_class.base_jacobian)
        self.ee_planner = CasadiPartialPlanner(self.qs, self.qs_dots, self.us, self.tfs, self.Ns, ee_config, self.motion_class.end_effector_pose, self.motion_class.compute_jacobian_whole)

    


