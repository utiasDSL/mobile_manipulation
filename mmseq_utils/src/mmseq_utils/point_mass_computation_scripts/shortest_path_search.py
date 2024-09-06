import sys

def dijkstra(graph, start):
    ''' Given a graph and a starting vertex, compute the shortest path to all other vertices in the graph. '''
    # Initialize distances and predecessors
    distances = {vertex: sys.maxsize for vertex in graph}
    predecessors = {vertex: None for vertex in graph}
    distances[start] = 0
    
    # Track visited vertices
    visited = set()
    
    while len(visited) < len(graph):
        # Choose the vertex with the shortest distance that hasn't been visited
        current_vertex = min((v for v in graph if v not in visited), key=lambda x: distances[x])
        
        # Mark current vertex as visited
        visited.add(current_vertex)
        
        # Update distances to adjacent vertices
        for neighbor, weight in graph[current_vertex].items():
            if neighbor not in visited:
                new_distance = distances[current_vertex] + weight[0][6]
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current_vertex
    
    return distances, predecessors

def shortest_path(graph, start, target):
    ''' Given a graph, a starting vertex, and a target vertex, compute the shortest path between the two vertices. '''
    distances, predecessors = dijkstra(graph, start)
    # Backtrack from target to start to construct the shortest path
    path = []
    current_vertex = target
    while current_vertex is not None:
        path.insert(0, current_vertex)
        current_vertex = predecessors[current_vertex]
    
    return distances[target], path

def shortest_path_multi_pose(graph, start_level, target_level, dict_poses):
    ''' Given a graph, a starting vertex, and a target vertex, compute the shortest path between the two vertices. '''
    distances, predecessors = dijkstra(graph, f'{start_level}')
    # Backtrack from target to start to construct the shortest path
    min_cost = sys.maxsize  
    for i in range(len(dict_poses[target_level])):
        current_vertex = f'{target_level}_{i}'
        path = []
        while current_vertex is not None:
            path.insert(0, current_vertex)
            current_vertex = predecessors[current_vertex]
        # check for min
        if distances[f'{target_level}_{i}'] < min_cost:
            min_cost = distances[f'{target_level}_{i}']
            min_path = path
    return min_cost, min_path