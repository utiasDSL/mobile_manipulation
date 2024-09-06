import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_velocities_in_cone_2d(points, module_min, module_max, vertex_angle_deg=30, magnitude_step=2, angle_step=20, loop=False, prediction_horizon=None):
    df = {}
    if not prediction_horizon:
        prediction_horizon = len(points)
    points_to_inspect = points[0:prediction_horizon+1]
    for i, point in enumerate(points_to_inspect):
        x, y = point
        # Convert vertex angle from degrees to radians
        vertex_angle_rad = np.deg2rad(vertex_angle_deg)

        # Generate all possible combinations of magnitude and angle
        magnitudes = np.linspace(module_min, module_max, magnitude_step)

        # Calculate the centerline vector for the current point
        if i < len(points_to_inspect) - 1:
            next_point = points_to_inspect[i + 1]
        else:
            if loop:
                next_point = points_to_inspect[0]  # Use initial point for the last point
            else:
                # next point is on the line between the last two points
                next_point = (2 * x - points_to_inspect[i - 1][0], 2 * y - points_to_inspect[i - 1][1])
        centerline_vector = np.array([next_point[0] - x, next_point[1] - y])

        # Filter velocities lying within the cone centered on the centerline vector
        cone_velocities = []
        for magnitude in magnitudes:
            # Calculate the angle of the velocity relative to the centerline vector
            centerline = np.arctan2(centerline_vector[1], centerline_vector[0])
            angle = centerline + np.linspace(-vertex_angle_rad, vertex_angle_rad, angle_step)
            vx = magnitude * np.cos(angle)
            vy = magnitude * np.sin(angle)
            if magnitude == 0 or i==0:
                vx = [0]
                vy = [0]
            
            #express velocities as points with x and y components
            combined_list = [(vx[i], vy[i]) for i in range(len(vx))]
            cone_velocities.extend(combined_list)
            # cone_velocities.append((vx, vy))
            if i==0:
                break   
        df[i] = cone_velocities
    return df

def compute_velocities_in_cone_3d(points, module_min, module_max, vertex_angle_deg=30, magnitude_step=2, angle_step=20, loop=False, prediction_horizon=None):
    df = {}
    if not prediction_horizon:
        prediction_horizon = len(points)
    points_to_inspect = points[0:prediction_horizon+1]
    for i, point in enumerate(points_to_inspect):
        x, y, z = point  # Extract x, y, z coordinates from point
        
        # Convert vertex angle from degrees to radians
        vertex_angle_rad = np.deg2rad(vertex_angle_deg)

        # Generate all possible combinations of magnitude and angle
        magnitudes = np.linspace(module_min, module_max, magnitude_step)

        # Calculate the centerline vector for the current point
        if i < len(points_to_inspect) - 1:
            next_point = points_to_inspect[i + 1]
        else:
            if loop:
                next_point = points_to_inspect[0]  # Use initial point for the last point
            else:
                # next point is on the line between the last two points
                next_point = (2 * x - points_to_inspect[i - 1][0], 
                              2 * y - points_to_inspect[i - 1][1], 
                              2 * z - points_to_inspect[i - 1][2])
        centerline_vector = np.array([next_point[0] - x, next_point[1] - y, next_point[2] - z])

        # Filter velocities lying within the cone centered on the centerline vector
        cone_velocities = []
        for magnitude in magnitudes:
            # Calculate the angle of the velocity relative to the centerline vector
            centerline_theta = np.arctan2(centerline_vector[1], centerline_vector[0])
            centerline_phi = np.arctan2(np.linalg.norm(centerline_vector[:2]), centerline_vector[2])
            theta = centerline_theta + np.linspace(-vertex_angle_rad, vertex_angle_rad, angle_step)
            phi = centerline_phi + np.linspace(-vertex_angle_rad, vertex_angle_rad, angle_step)
            combined_list = []
            if magnitude == 0 or i == 0:
                cone_velocities.append((0, 0, 0))
            else:
                for theta_val in theta:
                    for phi_val in phi:
                        vx = magnitude * np.cos(theta_val) * np.sin(phi_val)
                        vy = magnitude * np.sin(theta_val) * np.sin(phi_val)
                        vz = magnitude * np.cos(phi_val)
                        cone_velocities.append((vx, vy, vz))
            
            if i == 0:
                break

        df[i] = cone_velocities

    return df


def plot_velocity_cone_3d(df, points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Iterate through each point and its corresponding velocities
    for key, velocities in df.items():
        x, y, z = points[key]
        vx, vy, vz = zip(*velocities)  # Unzip the velocities into x, y, z components
        
        # Plot velocities in 3D
        ax.quiver([x]*len(vx), [y]*len(vy), [z]*len(vz), vx, vy, vz, length=0.1, normalize=True)

    # Set labels and show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Velocity Cone in 3D Space')
    plt.show()
