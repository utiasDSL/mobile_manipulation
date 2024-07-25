import casadi as ca
import numpy as np


def setup_single_problem(motion_class, start, goal, motion_model, forward_kinematic, N=100, d_tol=0.01, initial_point=False, state_dim=2, i=0, obstacles=[]):
    '''Function that generates the constraints needed to slove an NLP to minimize the time taken for a robot to move between two points'''
    # Check dimension of the points
    X = ca.MX.sym(f'X_{i}', state_dim*3, N)
    t = ca.MX.sym(f't_{i}', 1)
    dt = t/N

    total_elements = state_dim*3
    vel_start_index = state_dim
    u_start_index = 2*state_dim
    ee_point_dim = len(goal)/2

    q_min = motion_class.lb_x[:state_dim]
    q_max = motion_class.ub_x[:state]
    q_dot_min = motion_class.lb_x[state_dim:]
    q_dot_max = motion_class.ub_x[state_dim:]
    u_min = motion_class.lb_u
    u_max = motion_class.ub_u

    g = []
    lbg = []
    ubg = []
    lbx = []
    ubx = []
    
    for i in range(N-1):
        cur_lbx = ca.DM.ones(total_elements)*-np.inf
        cur_ubx = ca.DM.ones(total_elements)*np.inf
        # constraints on velocity
        cur_lbx[vel_start_index:u_start_index] = ca.horzcat(*q_dot_min)
        cur_ubx[vel_start_index:u_start_index] = ca.horzcat(*q_dot_max)
        # constraints on acceleration
        cur_lbx[u_start_index:] = ca.horzcat(*u_min)
        cur_ubx[u_start_index:] = ca.horzcat(*u_max)
        if i==0:
            if initial_point:
                cur_lbx[:vel_start_index] = start[:vel_start_index]
                cur_ubx[:vel_start_index] = start[:vel_start_index]
                if start[vel_start_index] is not None:
                    cur_lbx[vel_start_index:u_start_index] = start[vel_start_index:]
                    cur_ubx[vel_start_index:u_start_index] = start[vel_start_index:]
            else:
                # first point is the last of previous iteration so we need to set such identity
                g.append(X[:,i]-start)
                lbg.append(ca.DM.zeros(total_elements))
                ubg.append(ca.DM.zeros(total_elements))

        # Motion model
        g.append(X[:u_start_index, i+1] -  X[:u_start_index, i] - dt*motion_model(X[:u_start_index, i], X[u_start_index:, i]))
        lbg.append(ca.DM.zeros(2*state_dim))
        ubg.append(ca.DM.zeros(2*state_dim))
        lbx.append(cur_lbx)
        ubx.append(cur_ubx)
        # Obstacle constraints
        if len(obstacles) > 0:
            balls = motion_class.generate_balls_constraints(X[:vel_start_index, i])
            # Add floor constraints
            for i in range(1,3):
                g.append(balls[-i])
                lbg.append(ca.DM.zeros(2))
                ubg.append(ca.DM.ones(2)*np.inf)
        for obstacle in obstacles:
            for ball in balls:
                g.append(ca.norm_2(ball - ca.vertcat(obstacle.x, obstacle.y)) - obstacle.radius - motion_class.ball_radius)
                lbg.append(0)
                ubg.append(np.inf)
    
    # Final point constraints
    cur_lbx = ca.DM.ones(total_elements)*-np.inf
    cur_ubx = ca.DM.ones(total_elements)*np.inf
    g.append(forward_kinematic(X[:vel_start_index, N-1]) - goal)
    lbg.append(ca.DM.zeros(len(goal)))
    ubg.append(ca.DM.zeros(len(goal)))
    cur_lbx[vel_start_index:u_start_index] = ca.horzcat(*v_min)
    cur_ubx[vel_start_index:u_start_index] = ca.horzcat(*v_max)
    cur_lbx[u_start_index:] = ca.horzcat(*a_min)
    cur_ubx[u_start_index:] = ca.horzcat(*a_max)
    lbx.append(cur_lbx)
    ubx.append(cur_ubx)
    X = X.reshape((-1, 1))
    return X, g, lbg, ubg, lbx, ubx, t

               

def optimize_sequential(motion_class, points, prediction_horizon, X0, motion_model, forward_kinematic, Ns, d_tol=0.01, obstacles=[]):
    X = []
    g = []
    lbg = []
    ubg = []
    lbx = [0]*prediction_horizon #contains the time for each problem
    ubx = [np.inf]*prediction_horizon #contains the time for each problem
    ts = []
    state_dim = motion_class.DoF
    obj =0
    for i in range(prediction_horizon):
        is_initial_point = False
        if i == 0:
            is_initial_point = True
            start_point = X0[prediction_horizon:prediction_horizon+2*state_dim]
        else:
            start_point = X[-3*state_dim:]
        X_i, g_i, lbg_i, ubg_i, lbx_i, ubx_i, t_i = setup_single_problem(motion_class, start_point, points[i+1], motion_model, forward_kinematic, N=Ns[i], d_tol=d_tol, initial_point=is_initial_point, state_dim=state_dim, i=i, obstacles=obstacles)
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
    nlp = {'x': OPT_variables, 'f': obj, 'g': ca.vertcat(*g)}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    res = solver(x0=X0, lbx=ca.vertcat(*lbx), ubx=ca.vertcat(*ubx), lbg=ca.vertcat(*lbg), ubg=ca.vertcat(*ubg))
    X_dim = 3*state_dim
    result = res['x']
    final_time = ca.sum1(result[:prediction_horizon])
    X_final = ca.vertcat(final_time, result[prediction_horizon:])
    return X_final, X_dim
    



