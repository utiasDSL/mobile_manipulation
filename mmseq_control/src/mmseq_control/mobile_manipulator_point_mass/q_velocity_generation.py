import numpy as np

def generate_waypoints_q_dot(self, dict_graph, end_effector_velocity_dict, v_max, v_min):
    q_dot_dict = {}
    for i in range(len(dict_graph)):
        q_dot_dict[i] = {}
        for j in range(len(dict_graph[i])):
            q_dot_dict[i][j] = []
            for v0 in end_effector_velocity_dict[i]:
                current_pose = dict_graph[i][j]
                J = self.compute_jacobian_whole(current_pose)
                pseudo_inv_J = np.linalg.pinv(J)
                q_dot = pseudo_inv_J @ np.array([[v0[0]], [v0[1]], [v0[2]]])
                # Check if the velocity is within the limits
                append = True
                for k in range(len(v_max)):
                    if q_dot[k, 0] > v_max[k] or q_dot[k, 0] < v_min[k]:
                        append = False
                        break
                if append:
                    q_dot_dict[i][j].append(q_dot[:,0])
    return q_dot_dict

