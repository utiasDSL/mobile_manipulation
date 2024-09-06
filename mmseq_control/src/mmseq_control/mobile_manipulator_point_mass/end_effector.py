import numpy as np

def generate_intermediate_points_withouth_trajectory(self, points, prediction_horizon=None):
    intermediate_points = []
    if prediction_horizon == None:
        prediction_horizon = len(points)
    for i in range(prediction_horizon+1):
        if i == len(points):
            intermediate_points.append((points[0][0], points[0][1], points[0][2]))
        else:
            intermediate_points.append((points[i][0], points[i][1], points[i][2]))
    return intermediate_points