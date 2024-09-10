import sys
from .point_mass_trajectory_optimization import calculate_tf, compute_alpha
from .shortest_path_search import shortest_path, shortest_path_multi_pose

def compute_single_trajectory(start_point, end_point, a_max, a_min, v_max, v_min, starting_velocity, ending_velocity):
    ''' Given an initial 2D poind, a final 2D point, velocities and acceleration constraints, compute the x and y trajectory parameters. 
        Scale the trajectory with the lowest tf to match the other trajectory and return both trajectories. '''
    results = []
    for i in range(len(start_point)):
        # Compute trajectories
        trajectories = calculate_tf(start_point[i], starting_velocity[i], end_point[i], ending_velocity[i], a_max[i], a_min[i], v_max[i], v_min[i])
        if trajectories == None:
            return None      
        results.append(trajectories)
    times = []
    min_time_trajectories = [] 
    # Find the min element in the list of results
    for i in range(len(results)):
        min_time_trajectories.append(min(results[i], key=lambda x: x[6]))
        times.append(min(results[i], key=lambda x: x[6])[6])

    # Find the index of the largest element in the list of times
    tf = max(times)
    max_index = times.index(max(times))

    solution = ()
    # Extend the other trajectories 
    for i in range(len(results)):
        if i != max_index:
            extended_trajectory = compute_alpha(start_point[i], starting_velocity[i], end_point[i], ending_velocity[i], a_max[i], a_min[i], v_max[i], v_min[i], float(tf))
            if extended_trajectory == None:
                return None
            solution += (extended_trajectory[0],)
        else:
            solution += (min_time_trajectories[i],)

    return solution

def compute_trajectory_set(points, a_max, a_min, v_max, v_min, starting_velocity_list, ending_velocity_list, df, start_index, end_index, prediction_horizon):
    ''' Given two points, compute all possible trajectories between the two points using the provided velocity lists. Assemble the results in the dictionary used for the shortest path algorithm. 
    In such dictionary, all the intermediate graph nodes with the same starting coordinates but different starting velocities are cosidered as different states'''
    start_point = points[start_index]
    end_point= points[end_index]
    results = []
    for i, v0 in enumerate(starting_velocity_list):
        if start_index == 0 and i == 0:
                df[f'{start_index}'] = {}
        elif start_index !=0:
            df[f'{start_index}_{i}'] = {}
        for j, vf in enumerate(ending_velocity_list):
            params = compute_single_trajectory(start_point, end_point, a_max, a_min, v_max, v_min, v0, vf)
            if params == None:
                continue
            if start_index == 0 and prediction_horizon == 1:
                if (not f'{end_index}_f' in df[f'{start_index}'].keys()) or (params[0][6]<df[f'{start_index}'][f'{end_index}_f'][0][6]):
                    df[f'{start_index}'][f'{end_index}_f'] = params
                    df[f'{end_index}_f'] = {}
            elif start_index == 0:
                if (not f'{end_index}_{j}' in df[f'{start_index}'].keys()) or (params[0][6]<df[f'{start_index}'][f'{end_index}_{j}'][0][6]):
                    df[f'{start_index}'][f'{end_index}_{j}'] = params # create new state for every in between index
            elif start_index == prediction_horizon - 1:
                if (not f'{end_index}_f' in df[f'{start_index}_{i}'].keys()) or (params[0][6]<df[f'{start_index}_{i}'][f'{end_index}_f'][0][6]):
                    df[f'{start_index}_{i}'][f'{end_index}_f'] = params
                    df[f'{end_index}_f'] = {}
            else:
                if (not f'{end_index}_{j}' in df[f'{start_index}_{i}'].keys()) or (params[0][6]<df[f'{start_index}_{i}'][f'{end_index}_{j}'][0][6]):
                    df[f'{start_index}_{i}'][f'{end_index}_{j}'] = params
            results.append(params)
    return results, df

def compute_all_trajectories(points, velocity_df, max_acc, min_acc, max_vel, min_vel, prediction_horizon=None):
    ''' Given a list of points, compute all possible trajectories between the points using the provided acceleration and velocity constraints.'''
    results = []
    dict_res = {}

    if not prediction_horizon:
        prediction_horizon = len(points)
        
    points_to_inspect = points[0:prediction_horizon]
    for i in range(prediction_horizon):
        start_vel = velocity_df[i]
        if i == len(points) - 1:
            end_point = 0
            end_vel = velocity_df[0]
        else:
            end_point = i+1
            end_vel = velocity_df[i + 1]
        trajectories, dict_res = compute_trajectory_set(points, max_acc, min_acc, max_vel, min_vel, start_vel, end_vel, dict_res, i, end_point, prediction_horizon)
        results.extend(trajectories)

    if prediction_horizon == len(points):
        cost, shortest = shortest_path(dict_res, '0', '0_f')
    else:
        cost, shortest = shortest_path(dict_res, '0', f'{prediction_horizon}_f')
    return results, dict_res, shortest

def compute_trajectory_set_multi_pose(dict_poses, start_level, end_level, start_node, end_node,  a_max, a_min, v_max, v_min, starting_velocity_list, ending_velocity_list, result_dict, prediction_horizon):
    ''' Given two points, compute all possible trajectories between the two points using the provided velocity lists. Assemble the results in the dictionary used for the shortest path algorithm. 
    In such dictionary, all the intermediate graph nodes with the same starting coordinates but different starting velocities are cosidered as different states'''
    start_pose = dict_poses[start_level][start_node]
    end_pose = dict_poses[end_level][end_node]
    results = []
    
    for i, v0 in enumerate(starting_velocity_list):
        if start_level == 0 and i == 0 and not f'{start_level}' in result_dict.keys():
                result_dict[f'{start_level}'] = {}
        elif start_level !=0 and not f'{start_level}_{start_node}_{i}' in result_dict.keys():
            result_dict[f'{start_level}_{start_node}_{i}'] = {}
        for j, vf in enumerate(ending_velocity_list):
            params = compute_single_trajectory(start_pose, end_pose, a_max, a_min, v_max, v_min, v0, vf)
            if params == None:
                continue
            if start_level == 0 and prediction_horizon == 1:
                if (not f'{end_level}_{end_node}' in result_dict[f'{start_level}'].keys()) or (params[0][6]<result_dict[f'{start_level}'][f'{end_level}_{end_node}'][0][6]):
                    result_dict[f'{start_level}'][f'{end_level}_{end_node}'] = params
                    result_dict[f'{end_level}_{end_node}'] = {}
            elif start_level == 0:
                if (not f'{end_level}_{end_node}_{j}' in result_dict[f'{start_level}'].keys()) or (params[0][6]<result_dict[f'{start_level}'][f'{end_level}_{end_node}_{j}'][0][6]):
                    result_dict[f'{start_level}'][f'{end_level}_{end_node}_{j}'] = params # create new state for every in between index
            elif start_level == prediction_horizon - 1:
                if (not f'{end_level}_{end_node}' in result_dict[f'{start_level}_{start_node}_{i}'].keys()) or (params[0][6]<result_dict[f'{start_level}_{start_node}_{i}'][f'{end_level}_{end_node}'][0][6]):
                    result_dict[f'{start_level}_{start_node}_{i}'][f'{end_level}_{end_node}'] = params
                    result_dict[f'{end_level}_{end_node}'] = {}
            else:
                result_dict[f'{start_level}_{start_node}_{i}'][f'{end_level}_{end_node}_{j}'] = params
            results.append(params)
    return results, result_dict
    
def compute_all_trajectories_multi_pose(dict_poses, velocity_df, a_max, a_min, v_max, v_min, prediction_horizon=None):
    ''' Given a dictionary of poses, compute all possible trajectories between the poses using the provided acceleration and velocity constraints.'''
    results = []
    result_dict = {}
    if not prediction_horizon:
        prediction_horizon = len(dict_poses) -1
    for level in range(prediction_horizon):
        for start_node in range(len(dict_poses[level])):
            if level == len(dict_poses)-1:
                next_level = 0
            else:
                next_level = level+1
            for end_node in range(len(dict_poses[next_level])):
                starting_velocity_list = velocity_df[level][start_node]
                ending_velocity_list = velocity_df[next_level][end_node]

                trajectories, result_dict = compute_trajectory_set_multi_pose(dict_poses, level, next_level, start_node, end_node, a_max, a_min, v_max, v_min, starting_velocity_list, ending_velocity_list, result_dict, prediction_horizon)
                results.extend(trajectories)
    if prediction_horizon == len(dict_poses):
        cost, shortest = shortest_path_multi_pose(result_dict, 0, 0, dict_poses)
    else:
        cost, shortest = shortest_path_multi_pose(result_dict, 0,prediction_horizon, dict_poses)
    # print("shortest path: ", cost)
    return results, result_dict, shortest
