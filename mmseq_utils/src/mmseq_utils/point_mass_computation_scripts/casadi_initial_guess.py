import numpy as np  

from .point_mass_trajectory_optimization import space_curve, velocity_curve, acceleration_curve
from .trajectory_computations import compute_all_trajectories_multi_pose

def initial_guess(mobile_robot, points, starting_configurations, prediction_horizon, N, d_tol=0.01, lambda_num_cols=1, analyze=False, remove_bad=False, position_noise=0, velocity_noise=0, acceleration_noise=0, is_sequential_guess=False):
    results, dict_res, shortest = mobile_robot.calculate_trajectory(points, starting_configurations, prediction_horizon, magnitude_step=1)
    Ns, tf = compute_Ns(shortest, dict_res, N)
    ts = []
    for i in range(len(shortest)-1):
        t = dict_res[shortest[i]][shortest[i+1]][0][6]
        ts.append(t)
    X0_no_tn = []   

    for i in range(len(shortest)-1):
        # qs, q_dots, us = equally_spaced_in_place(points, i, shortest, dict_res, Ns[i])
        qs, q_dots, us = equally_spaced_in_time(i, shortest, dict_res, Ns[i])
        qs_reshaped = np.array(qs)
        q_dots_reshaped = np.array(q_dots)
        us_reshaped = np.array(us)
        # q_dots_reshaped = np.zeros(qs_reshaped.shape)
        # us_reshaped = np.zeros(qs_reshaped.shape)
        end_effector_pose_func = mobile_robot.end_effector_pose_func()
        for k in range(Ns[i]): 
            if k == 0 and i == 0:
                X0_no_tn.extend([*qs_reshaped[:,k], *q_dots_reshaped[:,k], *us_reshaped[:,k]])
            else:
                X0_no_tn.extend([*qs_reshaped[:,k]+position_noise*np.random.random(qs_reshaped[:,k].shape), *q_dots_reshaped[:,k]+velocity_noise*np.random.random(q_dots_reshaped[:,k].shape), *us_reshaped[:,k]+acceleration_noise*np.random.random(us_reshaped[:,k].shape)])

            # X0_no_tn.extend([1]*(3*lambda_num_cols))
            if not is_sequential_guess:
                # append lambdas
                if k == Ns[i]-1:
                    X0_no_tn.extend([0]*(i+1) + [1]*(prediction_horizon-i-1))
                else:
                    X0_no_tn.extend([0]*(i) + [1]*(prediction_horizon-i))
                # append nus and mus
                # compute distance to all points
                norms_array = []
                mus_array = []
                for j in range(1,prediction_horizon+1):
                    if j==len(points):
                        point = points[0]
                    else:
                        point = points[j]
                    p = np.array([point[0], point[1], point[2]])
                    norm = np.linalg.norm(end_effector_pose_func([*qs_reshaped[:,k]]) - p)
                    if norm <= d_tol:
                        mus_array.append(1)
                        norms_array.append(norm**2)
                    else:
                        mus_array.append(0)
                        norms_array.append(d_tol**2)
                # print('point:', x[k], y[k])
                # print(cpc(mus_array, [x[k], y[k], 0, 0], norms_array))
                X0_no_tn.extend(mus_array)
                X0_no_tn.extend(norms_array)                                                    
    if is_sequential_guess:
        return X0_no_tn, ts, Ns
    return X0_no_tn, tf


def equally_spaced_in_time(i, shortest, dict_res, N=100):
    params = dict_res[shortest[i]][shortest[i+1]]
    if i ==0:
        t_values = np.linspace(0, params[0][6], N)
    else:
        t_values = np.linspace(0, params[0][6], N+1) # this way, there will not be two overlapping points
        t_values = t_values[1:]

    qs = []
    q_dots = []
    us = []
    for i in range(len(params)):
        qs.append([space_curve(t, params[i][0], params[i][1], params[i][2], params[i][3], params[i][4], params[i][5], params[i][7], params[i][8]) for t in t_values])
        q_dots.append([velocity_curve(t, params[i][1], params[i][2], params[i][3], params[i][4], params[i][5], params[i][7], params[i][8]) for t in t_values])
        us.append([acceleration_curve(t, params[i][2], params[i][5], params[i][7], params[i][8]) for t in t_values])
    return qs, q_dots, us

def equally_spaced_in_place(points, i, shortest, dict_res, N=100):
    qs_bad, q_dots, us = equally_spaced_in_time(i, shortest, dict_res, N)

    qs = []
    params = dict_res[shortest[i]][shortest[i+1]]
    for i in range(len(qs_bad)):
        starting_point = space_curve(0, params[i][0], params[i][1], params[i][2], params[i][3], params[i][4], params[i][5], params[i][7], params[i][8])
        ending_point = space_curve(params[i][6], params[i][0], params[i][1], params[i][2], params[i][3], params[i][4], params[i][5], params[i][7], params[i][8])
        qs.append(np.linspace(starting_point, ending_point, N))

    return qs, q_dots, us

def compute_tf(shortest, dict_res):
    tf = 0
    for i in range(len(shortest)-1):
        tf += dict_res[shortest[i]][shortest[i+1]][0][6]
    return tf

def compute_Ns(shortest, dict_res, N):
    Ns = []

    tf = compute_tf(shortest, dict_res)
    ts = np.linspace(0, tf, N)
    t_pref = 0
    for i in range(len(shortest)-1):
        t = dict_res[shortest[i]][shortest[i+1]][0][6]
        # find how many points to sample from t_pref to t
        N_to_append = int(N*(t-t_pref)/tf)
        if i == len(shortest)-2:
            N_to_append = N - sum(Ns)
        Ns.append(N_to_append)
    return Ns, tf 

def initial_guess_simple(mobile_robot, points, starting_configurations, prediction_horizon, N, is_sequential_guess=False, d_tol=0.01):
    waypoints_poses, intermediate_points = mobile_robot.calculate_trajectory_simple(points, starting_configurations, prediction_horizon)

    # Use cartesian distance between points to determine sampling rate
    total_distance = 0
    distances = []
    for i in range(prediction_horizon):
        d = np.linalg.norm(np.array(points[i]) - np.array(points[i+1]))
        total_distance += d
        distances.append(d)
    
    # Assemble the initial guess
    X0_no_tn = []
    Ns = []
    end_effector_pose_func = mobile_robot.end_effector_pose_func()
    ts = []
    for i in range(prediction_horizon):
        n = int(N*distances[i]/total_distance)

        if i == prediction_horizon-1:
            n = N - sum(Ns) -1 
        Ns.append(n)


        # sample n-1 points from intermediate_points[i]
        controller_times = intermediate_points[i][0]
        total_segment_time = controller_times[-1]
        ts.append(total_segment_time)
        interpolation_times = np.linspace(0, total_segment_time, n)
        qs = np.array(intermediate_points[i][1]).T
        q_dots = np.array(intermediate_points[i][2]).T
        interpolated_qs = np.array([np.interp(interpolation_times, controller_times, q) for q in qs]).T

        interpolated_q_dots = np.array([np.interp(interpolation_times, controller_times, q_dot) for q_dot in q_dots]).T
        for k in range(n):
            X0_no_tn.extend(interpolated_qs[k]) # qs
            X0_no_tn.extend(interpolated_q_dots[k]) # q_dots
            # X0_no_tn.extend([0]*mobile_robot.DoF) # q_dots
            X0_no_tn.extend([0]*mobile_robot.DoF) # us

            if not is_sequential_guess:
                # append lambdas
                if k == n-1:
                    X0_no_tn.extend([0]*(i+1) + [1]*(prediction_horizon-i-1))
                else:
                    X0_no_tn.extend([0]*(i) + [1]*(prediction_horizon-i))
                # append nus and mus
                # compute distance to all points
                norms_array = []
                mus_array = []
                for j in range(1,prediction_horizon+1):
                    if j==len(points):
                        point = points[0]
                    else:
                        point = points[j]
                    p = np.array([point[0], point[1], point[2]])
                    norm = np.linalg.norm(end_effector_pose_func(interpolated_qs[k]) - p)
                    if norm <= d_tol:
                        mus_array.append(1)
                        norms_array.append(norm**2)
                    else:
                        mus_array.append(0)
                        norms_array.append(d_tol**2)

                X0_no_tn.extend(mus_array)
                X0_no_tn.extend(norms_array)    
    # add the last point  
    X0_no_tn.extend(waypoints_poses[-1]) # qs
    X0_no_tn.extend([0]*mobile_robot.DoF) # q_dots
    X0_no_tn.extend([0]*mobile_robot.DoF) # us
    if not is_sequential_guess:
        # append lambdas 
        X0_no_tn.extend([0]*(prediction_horizon))
        # append mus 
        X0_no_tn.extend([0]*(prediction_horizon))
        # append nus
        X0_no_tn.extend([0]*(prediction_horizon))
    Ns[-1] += 1
    if is_sequential_guess:
        return X0_no_tn, ts, Ns
    return X0_no_tn, np.sum(ts)

def slow_down_guess(start_state, end_state, N, mobile_robot, position_noise=0, velocity_noise=0, acceleration_noise=0, a_bounds=None):
    # create pose dict
    waypoints_poses = []
    waypoints_poses.append(start_state[:mobile_robot.DoF].full().flatten())
    waypoints_poses.append(end_state[:mobile_robot.DoF].full().flatten())
    dict_graph = mobile_robot.create_graph(waypoints_poses)

    # create dictionary of results
    velocities = {}
    velocities[0] = {}
    velocities[1] = {}
    velocities[0][0] = [start_state[mobile_robot.DoF:2*mobile_robot.DoF].full().flatten()]
    velocities[1][0] = [end_state[mobile_robot.DoF:2*mobile_robot.DoF].full().flatten()]

    # set accelerations
    if a_bounds is None:
        a_max = mobile_robot.ub_u
        a_min = mobile_robot.lb_u
    else:
        a_min = a_bounds[0]
        a_max = a_bounds[1]
    v_max = mobile_robot.ub_x[mobile_robot.DoF:]
    v_min = mobile_robot.lb_x[mobile_robot.DoF:]

    # compute trajectories
    results, dict_res, shortest = compute_all_trajectories_multi_pose(dict_graph, velocities, a_max, a_min, v_max, v_min, prediction_horizon=1)

    X0_no_tn = []   
    tf = results[0][0][6]

    qs, q_dots, us = equally_spaced_in_time(0, shortest, dict_res, N)
    qs_reshaped = np.array(qs)
    q_dots_reshaped = np.array(q_dots)
    us_reshaped = np.array(us)
    # q_dots_reshaped = np.zeros(qs_reshaped.shape)
    # us_reshaped = np.zeros(qs_reshaped.shape)
    end_effector_pose_func = mobile_robot.end_effector_pose_func()
    for k in range(N): 
        if k == 0:
            X0_no_tn.extend([*qs_reshaped[:,k], *q_dots_reshaped[:,k], *us_reshaped[:,k]])
        else:
            X0_no_tn.extend([*qs_reshaped[:,k]+position_noise*np.random.random(qs_reshaped[:,k].shape), *q_dots_reshaped[:,k]+velocity_noise*np.random.random(q_dots_reshaped[:,k].shape), *us_reshaped[:,k]+acceleration_noise*np.random.random(us_reshaped[:,k].shape)])

    return X0_no_tn, tf

def slow_down_guess_only_ee(start_state, ee_goal, N, mobile_robot, position_noise=0, velocity_noise=0, acceleration_noise=0, a_bounds=None):
    # find configuration that reaches the end effector goal
    starting_ee = mobile_robot.end_effector_pose(start_state[:mobile_robot.DoF].full().flatten())
    waypoints_poses, _ = mobile_robot.compute_positions(start_state[:mobile_robot.DoF].full().flatten(), [starting_ee, ee_goal])
    dict_graph = mobile_robot.create_graph(waypoints_poses)
    print('waypoints_poses:', waypoints_poses)

    for i in range(1, len(waypoints_poses)):
        for j in range(len(waypoints_poses[i])):
            if abs(waypoints_poses[i][j] - waypoints_poses[i-1][j])<0.000001:
                waypoints_poses[i][j] += 0.00001

    # create dictionary of results
    velocities = {}
    velocities[0] = {}
    velocities[1] = {}
    velocities[0][0] = [start_state[mobile_robot.DoF:2*mobile_robot.DoF].full().flatten()]
    velocities[1][0] = [np.zeros(mobile_robot.DoF)]

    # set accelerations
    if a_bounds is None:
        a_max = mobile_robot.ub_u
        a_min = mobile_robot.lb_u
    else:
        a_min = a_bounds[0]
        a_max = a_bounds[1]
    v_max = mobile_robot.ub_x[mobile_robot.DoF:]
    v_min = mobile_robot.lb_x[mobile_robot.DoF:]

    # compute trajectories
    results, dict_res, shortest = compute_all_trajectories_multi_pose(dict_graph, velocities, a_max, a_min, v_max, v_min, prediction_horizon=1)

    X0_no_tn = []   
    tf = results[0][0][6]

    qs, q_dots, us = equally_spaced_in_time(0, shortest, dict_res, N)
    qs_reshaped = np.array(qs)
    q_dots_reshaped = np.array(q_dots)
    us_reshaped = np.array(us)
    # q_dots_reshaped = np.zeros(qs_reshaped.shape)
    # us_reshaped = np.zeros(qs_reshaped.shape)
    end_effector_pose_func = mobile_robot.end_effector_pose_func()
    for k in range(N): 
        if k == 0:
            X0_no_tn.extend([*qs_reshaped[:,k], *q_dots_reshaped[:,k], *us_reshaped[:,k]])
        else:
            X0_no_tn.extend([*qs_reshaped[:,k]+position_noise*np.random.random(qs_reshaped[:,k].shape), *q_dots_reshaped[:,k]+velocity_noise*np.random.random(q_dots_reshaped[:,k].shape), *us_reshaped[:,k]+acceleration_noise*np.random.random(us_reshaped[:,k].shape)])

    return X0_no_tn, tf

