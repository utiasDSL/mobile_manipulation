import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from mmseq_utils.plot_casadi_time_optimal import decompose_X
from mmseq_utils.point_mass_computation_scripts.casadi_initial_guess import initial_guess, initial_guess_simple
from mmseq_control.mobile_manipulator_point_mass.mobile_manipulator_class import MobileManipulatorPointMass
from mmseq_plan.PlanBaseClass import WholeBodyPlanner, CasadiPartialPlanner


class CPCPlanner(WholeBodyPlanner):
    def __init__(self, motion_class):
        self.motion_class = motion_class
        self.qs_num = motion_class.DoF
        self.q_max = motion_class.ub_x[:self.qs_num]
        self.q_min = motion_class.lb_x[:self.qs_num]
        self.q_dot_max = motion_class.ub_x[self.qs_num:]
        self.q_dot_min = motion_class.lb_x[self.qs_num:]
        self.u_max = motion_class.ub_u
        self.u_min = motion_class.lb_u
        self.motion_model = motion_class.ssSymMdl["fmdl"]
        self.forward_kinematic = motion_class.end_effector_pose_func()
        self.config = None

    def initializeFromMotionClass(self, motion_class):
        self.motion_class = motion_class
        self.qs_num = motion_class.DoF
        self.q_max = motion_class.ub_x[:self.qs_num]
        self.q_min = motion_class.lb_x[:self.qs_num]
        self.q_dot_max = motion_class.ub_x[self.qs_num:]
        self.q_dot_min = motion_class.lb_x[self.qs_num:]
        self.u_max = motion_class.ub_u
        self.u_min = motion_class.lb_u
        self.motion_model = motion_class.ssSymMdl["fmdl"]
        self.forward_kinematic = motion_class.end_effector_pose_func()
        self.config = None


    def optimize(self, points, prediction_horizon, X0, t_bound, motion_model, forward_kinematic, N=100, d_tol=0.01, constrain_final_point=False, cpc_tolerance=0.001, obstacles_avoidance=None):
        ''' Given a list of points, compute all possible trajectories between the points using the provided acceleration and velocity constraints.'''
        # _____SETUP CASADI PROBLEM_____

        # Number of degrees of freedom

        x_dyn = ca.MX.sym('x_dyn', 2*self.qs_num) # x_dyn looks like [x0, x1, v0, v1], column vector
        x_dyn_num_cols = x_dyn.size()[0]
        state_dim = self.qs_num

        control = ca.MX.sym('control', self.qs_num) # control looks like [u0, u1], column vector
        control_num_cols = control.size()[0]
        lambda_num_cols = prediction_horizon #assume starting point is not waypoint
        total_elements = 3*self.qs_num + (3)*(prediction_horizon)

        x_sample = ca.MX.sym('x_sample', total_elements) # x_sample looks like [x_dyn, u0, u1, lamdas, mus, vs]

        X = ca.MX.sym('X', total_elements, N) # X looks like [[x_0], ... , [x_N]] where x_i = [x_i, v_i, u_i, lambda_i, mu_i, v_i] 
        tn = ca.MX.sym('tn')
        dt = tn/(N)
        P = ca.MX.sym('P', 2*self.qs_num+len(points[0])+1) # P looks like [x_0, v_0, u_0, lambda_0, mu_0, v_0]

        vel_start_index = state_dim 
        u_start_index = x_dyn_num_cols 
        u_end_index = u_start_index + control_num_cols
        lambda_start_index = u_end_index
        mu_start_index = lambda_start_index + prediction_horizon  
        nu_start_index = mu_start_index + prediction_horizon  

        obj = 0
        # Define the objective function, calculate the norm of all us in X
        obj += tn

        # test the motion model
        # print(motion_model(ca.DM.ones(x_dyn_num_cols)*2, ca.DM.zeros(control_num_cols)))

        # Define CPC condition
        mus = ca.MX.sym('mus', lambda_num_cols)
        vs = ca.MX.sym('vs', lambda_num_cols)
        # need mus, vs, norms
        # first calculate distance of x_dyn to all points in points, make function out of it
        norms_array = []
        for i in range(1,prediction_horizon+1):
            if i==len(points):
                point = points[0]
            else:
                point = points[i]
            # print('point:', point)
            norms_array.append(ca.norm_2(forward_kinematic(x_dyn[:state_dim]) - ca.vertcat(*point))**2)

        distances = ca.Function('distances', [x_dyn, vs], [ca.vertcat(*norms_array) - vs])

        # Final function
        rhs_cpc = mus * distances(x_dyn, vs)
        cpc = ca.Function('cpc', [mus, x_dyn, vs], [rhs_cpc])

        # Define function to help calculating lambda constraints
        lambdas = ca.MX.sym('lambdas', lambda_num_cols)
        def compute_difference_lambdas(vec):
            diff = []
            for i in range(vec.size()[0]-1):
                diff.append(vec[i] - vec[i+1])
            return ca.vertcat(*diff)

        diff_lambdas = ca.Function('diff_lambdas', [lambdas], [compute_difference_lambdas(lambdas)])

        # test cpc
        # print("starting")
        # mu_test = ca.DM.ones(lambda_num_cols)
        # v_test = ca.DM.ones(lambda_num_cols)
        # x_test = ca.DM.ones(2*len(v_max))
        # print(distances(x_test, v_test))
        # print(cpc(mu_test, x_test, v_test))


        # Define the constraints
        g = []
        lbg = []
        ubg = []
        lbx = [0] # initial time
        ubx = [] # final time inserted later
        x_zero = X0[1:x_dyn_num_cols+1]
        xf = ca.vertcat(*points[prediction_horizon])
        for k in range(N): # N is the number of columns
            cur_lbx = ca.DM.ones(total_elements)*-np.inf
            cur_ubx = ca.DM.ones(total_elements)*np.inf 
            # constraints on position
            cur_lbx[:vel_start_index] = ca.horzcat(*self.q_min)
            cur_ubx[:vel_start_index] = ca.horzcat(*self.q_max)
            # constraints on velocity
            cur_lbx[vel_start_index:u_start_index] = ca.horzcat(*self.q_dot_min)
            cur_ubx[vel_start_index:u_start_index] = ca.horzcat(*self.q_dot_max)
            # constraints on acceleration
            cur_lbx[u_start_index:u_end_index] = ca.horzcat(*self.u_min)
            cur_ubx[u_start_index:u_end_index] = ca.horzcat(*self.u_max)
            if k==0:
                # Initial state
                g.append(X[:u_start_index, k] - x_zero)
                lbg.append(ca.DM.zeros(x_dyn_num_cols))
                ubg.append(ca.DM.zeros(x_dyn_num_cols))
                # initial lambda
                cur_lbx[lambda_start_index:mu_start_index] = ca.DM.ones(lambda_num_cols)
                cur_ubx[lambda_start_index:mu_start_index] = ca.DM.ones(lambda_num_cols)
            else:
                # Motion model
                g.append(X[:u_start_index, k] -  X[:u_start_index, k-1] - dt*motion_model(X[:u_start_index, k-1], X[u_start_index:u_end_index, k-1]))
                lbg.append(ca.DM.zeros(x_dyn_num_cols))
                ubg.append(ca.DM.zeros(x_dyn_num_cols))
                # lambda constraints
                if k == N-1: #last point
                    # Final lambda
                    cur_lbx[lambda_start_index:mu_start_index] = ca.DM.zeros(lambda_num_cols)
                    cur_ubx[lambda_start_index:mu_start_index] = ca.DM.zeros(lambda_num_cols)
                    # xf constraints
                    if constrain_final_point:
                        g.append(forward_kinematic(X[:vel_start_index, k]) - xf)
                        lbg.append(ca.DM.zeros(len(points[0])))
                        ubg.append(ca.DM.zeros(len(points[0])))
                    
                # lambda lex order
                g.append(diff_lambdas(X[lambda_start_index:mu_start_index, k]))
                lbg.append(ca.DM.ones(lambda_num_cols-1)*(-np.inf))
                ubg.append(ca.DM.zeros(lambda_num_cols-1))
                # progress lambda
                g.append(X[lambda_start_index:mu_start_index, k] - X[lambda_start_index:mu_start_index, k-1] + X[mu_start_index: nu_start_index, k-1])
                lbg.append(ca.DM.zeros(lambda_num_cols))
                ubg.append(ca.DM.zeros(lambda_num_cols))
    
            # mu constraints
            cur_lbx[mu_start_index: nu_start_index] = ca.DM.zeros(lambda_num_cols)
            cur_ubx[mu_start_index: nu_start_index] = ca.DM.ones(lambda_num_cols) # bound by one as per code
            # nu constraints all vs are between zero and d_tol**2
            cur_lbx[nu_start_index:] = ca.DM.zeros(lambda_num_cols)
            cur_ubx[nu_start_index:] = ca.DM.ones(lambda_num_cols)*d_tol**2
            # CPC constraints
            g.append(cpc(X[mu_start_index: nu_start_index, k], X[:u_start_index, k], X[nu_start_index:, k]))
            lbg.append(ca.DM.zeros(lambda_num_cols))
            ubg.append(ca.DM.ones(lambda_num_cols)*cpc_tolerance)
            # Self avoidance
            if obstacles_avoidance is not None:
                obs = obstacles_avoidance(X[:vel_start_index, k])
                g.append(obs)
                lbg.append(ca.DM.zeros(obs.size()[0]))
                ubg.append(np.inf*ca.DM.ones(obs.size()[0]))
                    
            # Append to lbx and ubx
            lbx.append(cur_lbx)
            ubx.append(cur_ubx)
    

        opts = {'print_time': False, 'ipopt.sb': 'yes', 'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6, 'ipopt.print_level': 0, 'ipopt.max_iter': 10000}
        OPT_variables = ca.vertcat(tn, X.reshape((-1, 1)))

        nlp = {'x': OPT_variables, 'f': obj, 'g': ca.vertcat(*g)}
        
        # nlp = {'x': OPT_variables, 'f': obj, 'g': , 'p': P}
        
    
        # create empty X0_no_tn
        # X0_no_tn = np.zeros(len(X0_no_tn))
        # _____SOLVE THE PROBLEM_____
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        args = {'lbx': ca.vertcat(*lbx), 'ubx': ca.vertcat(t_bound,*ubx), 'lbg': ca.vertcat(*lbg), 'ubg': ca.vertcat(*ubg)}
        # args = {'lbx': ca.vertcat(*lbx), 'ubx': ca.vertcat(*ubx), 'lbg': 0, 'ubg': 0}

        res = solver(x0=X0, **args)
        X = res['x']
        # print(X[0])
        if solver.stats()['return_status'] != 'Solve_Succeeded':
            print(solver.stats()['return_status'])

        return X, total_elements

    def generateTrajectory(self, points, starting_configuration, prediction_horizon, N=100, t_bound=np.inf, d_tol=0.01, cpc_tolerance=0.001, simple=False, obs_avoidance=True):
        self.N = N
        obstacles_avoidance = None
        if obs_avoidance:
            obstacles_avoidance = self.motion_class.casadi_model_interface.signedDistanceSymMdlsPerGroup["self"]
        points_full = [self.motion_class.end_effector_pose(starting_configuration).full().flatten()]
        points_full.extend(points)
        if simple:
            X0_array, tf = initial_guess_simple(self.motion_class, points_full, starting_configuration, prediction_horizon, N)
        else:
            X0_array, tf = initial_guess(self.motion_class, points_full, starting_configuration, prediction_horizon, N, d_tol=d_tol)
        X0 = ca.vertcat(tf, *X0_array)
        X, total_elements = self.optimize(points_full, prediction_horizon, X0, t_bound, self.motion_model, self.forward_kinematic, N=N, d_tol=d_tol, cpc_tolerance=cpc_tolerance, obstacles_avoidance=obstacles_avoidance)
        
        self.X = X
        self.total_elements = total_elements
        return X, total_elements
    
    def processResults(self):
        self.dt = float(self.X[0]/self.N)
        qs, q_dots, us = decompose_X(self.X, self.qs_num, self.total_elements)
        self.qs, self.qs_dots, self.us = qs.T, q_dots.T, us.T
        self.tf = float(self.X[0])

    def interpolate(self, t):
        ''' Given a time, return the interpolated q and q_dot'''
        lower_index = int(t/self.dt)
        upper_index = lower_index + 1
        if upper_index >= len(self.qs_dots):
            q = self.qs[-1]
            return q, [0]*self.qs_num
        lower_time = lower_index*self.dt
        upper_time = upper_index*self.dt
        q = self.qs[lower_index] + (self.qs[upper_index] - self.qs[lower_index])*(t - lower_time)/(upper_time - lower_time)
        q_dot = self.qs_dots[lower_index] + (self.qs_dots[upper_index] - self.qs_dots[lower_index])*(t - lower_time)/(upper_time - lower_time)
            # q.append(self.qs[i][lower_index] + (self.qs[i][upper_index] - self.qs[i][lower_index])*(t - lower_time)/(upper_time - lower_time))
            # q_dot.append(self.qs_dots[i][lower_index] + (self.qs_dots[i][upper_index] - self.qs_dots[i][lower_index])*(t - lower_time)/(upper_time - lower_time))
        return q, q_dot

    def initiliazePartialPlanners(self):
        if self.config is None:
            base_config = {"name": "base_planner", "type": "base", "tracking_err_tol": 0.02, "frame_id": "base", "ref_data_type": "Vec2"}
            ee_config = {"name": "ee_planner", "type": "end_effector", "tracking_err_tol": 0.02, "frame_id": "EE", "ref_data_type": "Vec3"}

        else:
            base_config = {"name": "base_planner", "type": "base", "tracking_err_tol": self.config["tracking_err_tol"], "frame_id": "base", "ref_data_type": "Vec2"}
            ee_config = {"name": "ee_planner", "type": "end_effector", "tracking_err_tol": self.config["tracking_err_tol"], "frame_id": "EE", "ref_data_type": "Vec3"}

        self.base_planner =  CPCPartialPlanner(self.X, self.N, self.total_elements, base_config, self.qs_num, self.motion_class.base_xyz, self.motion_class.base_jacobian)
        self.ee_planner = CPCPartialPlanner(self.X, self.N, self.total_elements, ee_config, self.qs_num, self.motion_class.end_effector_pose, self.motion_class.compute_jacobian_whole)



class CPCPartialPlanner(CasadiPartialPlanner):
    def __init__(self, X, N, total_elements, config, qs_num, pose_calculator_func, jacobian_calculator_func):
        super().__init__(config, pose_calculator_func, jacobian_calculator_func)
        self.X = X
        self.N = N
        self.total_elements = total_elements
        self.qs_num = qs_num

    def processResults(self):
        self.dt = float(self.X[0]/self.N)
        qs, q_dots, us = decompose_X(self.X, self.qs_num, self.total_elements)
        self.qs, self.qs_dots, self.us = qs.T, q_dots.T, us.T
        self.tf = float(self.X[0])
    
    def interpolate(self, t):
        ''' Given a time, return the interpolated q and q_dot'''
        lower_index = int(t/self.dt)
        upper_index = lower_index + 1
        if upper_index >= len(self.qs_dots):
            q = self.qs[-1]
            return q, [0]*self.qs_num
        lower_time = lower_index*self.dt
        upper_time = upper_index*self.dt
        q = self.qs[lower_index] + (self.qs[upper_index] - self.qs[lower_index])*(t - lower_time)/(upper_time - lower_time)
        q_dot = self.qs_dots[lower_index] + (self.qs_dots[upper_index] - self.qs_dots[lower_index])*(t - lower_time)/(upper_time - lower_time)
            # q.append(self.qs[i][lower_index] + (self.qs[i][upper_index] - self.qs[i][lower_index])*(t - lower_time)/(upper_time - lower_time))
            # q_dot.append(self.qs_dots[i][lower_index] + (self.qs_dots[i][upper_index] - self.qs_dots[i][lower_index])*(t - lower_time)/(upper_time - lower_time))
        return q, q_dot 
