import numpy as np

from mmseq_utils.point_mass_computation_scripts.velocity_generation import compute_velocities_in_cone_3d, plot_velocity_cone_3d
from mmseq_utils.point_mass_computation_scripts.trajectory_computations import compute_all_trajectories_multi_pose


def calculate_trajectory(self, points, starting_configurations, prediction_horizon,vertex_angle_deg=180,magnitude_step=1, angle_step=20, loop=False, sampling_rate=3):

    a_max = self.ub_u
    a_min = self.lb_u
    v_max = self.ub_x[self.DoF:]
    v_min = self.lb_x[self.DoF:]

    int_points = self.generate_intermediate_points_withouth_trajectory(points, prediction_horizon=prediction_horizon)
    waypoints_poses, _ = self.compute_positions(starting_configurations, int_points)

    for i in range(1, len(waypoints_poses)):
        for j in range(len(waypoints_poses[i])):
            if abs(waypoints_poses[i][j] - waypoints_poses[i-1][j])<0.000001:
                waypoints_poses[i][j] += 0.00001

    if prediction_horizon == None:
        points.append(points[0])

    max_vel_mag = np.linalg.norm(v_max)/2
    dict_graph = self.create_graph(waypoints_poses)
    end_effector_velocities = compute_velocities_in_cone_3d(points, 0, max_vel_mag, vertex_angle_deg=vertex_angle_deg, magnitude_step=magnitude_step, angle_step=angle_step, loop=loop, prediction_horizon=prediction_horizon)
    # plot_velocity_cone_3d(end_effector_velocities, points)
    q_dot_dict = self.generate_waypoints_q_dot(dict_graph, end_effector_velocities, v_max, v_min)
    results, dict_res, shortest = compute_all_trajectories_multi_pose(dict_graph, q_dot_dict, a_max, a_min, v_max, v_min, prediction_horizon=prediction_horizon)
    return results, dict_res, shortest

def calculate_trajectory_simple(self, points, starting_configurations, prediction_horizon, vertex_angle_deg=180,magnitude_step=4, angle_step=20, loop=False, sampling_rate=3):

    int_points = self.generate_intermediate_points_withouth_trajectory(points, prediction_horizon=prediction_horizon)
    
    waypoints_poses, intermediate_points = self.compute_positions(starting_configurations, int_points)
    return waypoints_poses, intermediate_points