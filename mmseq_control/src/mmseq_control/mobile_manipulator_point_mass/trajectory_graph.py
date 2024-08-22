
def create_graph(self, waypoints_poses, end_effector_goals, sampling_rate=3):
    dict_graph = {}
    dict_graph[0] = [waypoints_poses[0]]
    for i in range(1,len(waypoints_poses)):
        sampled_poses = [waypoints_poses[i]]
        dict_graph[i] = sampled_poses
    return dict_graph

        