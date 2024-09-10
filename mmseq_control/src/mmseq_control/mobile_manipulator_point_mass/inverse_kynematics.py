import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def inverse_kynematics_dynamic(self, current_end_effector, current_params, future_goals, k =1, t=0.1):
    ''' Compute the inverse kinematics for the mobile manipulator. '''
    # Compute the inverse kinematics for the mobile manipulator

    qs, intermediate_qs, intermediate_qs_dots = self.new_pose_luoponov_control(current_params, future_goals[0], t=t)

    return qs, intermediate_qs, intermediate_qs_dots

def new_pose_luoponov_control(self, current_params, goal, t=0.1, max_iter=100):
    e, J = self.compute_error(current_params, goal)
    qs = np.array(current_params).reshape(-1, 1)
    intermediate_qs = [qs.reshape(-1)]
    intermediate_qs_dots = [(0,)*self.DoF]
    count = 0
    while np.linalg.norm(e) > 0.01 and count < max_iter:
        J_inv = np.linalg.pinv(J)
        q_dot = J_inv @ (e)
        qs = qs + t*q_dot[:, 0]
        intermediate_qs.append(qs.full().flatten())
        intermediate_qs_dots.append(q_dot.full().flatten())
        e, J = self.compute_error(qs, goal)
        count += 1
    if count == max_iter:
        return [None]*self.DoF
    return qs, intermediate_qs, intermediate_qs_dots

def compute_error(self, current_params, future_goal):
    ''' Compute the error for the inverse kinematics. '''
    # Compute end effector position
    ee_coord = self.end_effector_pose(current_params)
    # Compute Jacobian
    J = self.compute_jacobian_whole(current_params)
    # compute inverse of Jacobian
    e = future_goal[:3] - ee_coord
    return e, J

def compute_positions(self, starting_configurations, intermediate_points, t_spacing=0.1):
    starting_configuration_copy = starting_configurations.copy()
    waypoints_configs = [starting_configuration_copy]
    controller = []
    qs = starting_configuration_copy    
    for i in range(1,len(intermediate_points)):
        qs, intermediate_qs, intermediate_qs_dots = self.inverse_kynematics_dynamic(intermediate_points[i-1], qs, intermediate_points[i:], t=t_spacing)
        inter_times = np.linspace(0, len(intermediate_qs)*t_spacing, len(intermediate_qs))
        # print('qs',qs.full().flatten())
        waypoints_configs.append(qs.full().flatten())
        controller.append([inter_times, intermediate_qs, intermediate_qs_dots])
    return waypoints_configs, controller