import numpy as np
import casadi as ca

from mmseq_control.robot import MobileManipulator3D, CasadiModelInterface


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles to a 3D rotation matrix (SO(3)) using CasADi.
    """
    R_x = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, ca.cos(roll), -ca.sin(roll)),
        ca.horzcat(0, ca.sin(roll), ca.cos(roll))
    )
    
    R_y = ca.vertcat(
        ca.horzcat(ca.cos(pitch), 0, ca.sin(pitch)),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-ca.sin(pitch), 0, ca.cos(pitch))
    )
    
    R_z = ca.vertcat(
        ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0),
        ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0),
        ca.horzcat(0, 0, 1)
    )

    # Combine the rotation matrices around x, y, and z axes
    R = ca.mtimes(R_z, ca.mtimes(R_y, R_x))
    return R

class MobileManipulatorPointMass(MobileManipulator3D):
    def __init__(self, config):

        self.name = 'MobileManipulatorPointMass'

        from . import inverse_kynematics
        setattr(MobileManipulatorPointMass, 'inverse_kynematics_dynamic', inverse_kynematics.inverse_kynematics_dynamic)
        setattr(MobileManipulatorPointMass, 'compute_positions', inverse_kynematics.compute_positions)
        setattr(MobileManipulatorPointMass, 'compute_error', inverse_kynematics.compute_error)
        setattr(MobileManipulatorPointMass, 'new_pose_luoponov_control', inverse_kynematics.new_pose_luoponov_control)
        from . import end_effector
        setattr(MobileManipulatorPointMass, 'generate_intermediate_points_withouth_trajectory', end_effector.generate_intermediate_points_withouth_trajectory)
        from . import trajectory_graph
        setattr(MobileManipulatorPointMass, 'create_graph', trajectory_graph.create_graph)
        from . import q_velocity_generation
        setattr(MobileManipulatorPointMass, 'generate_waypoints_q_dot', q_velocity_generation.generate_waypoints_q_dot)
        from . import motion_model
        setattr(MobileManipulatorPointMass, 'robot_motion_model', motion_model.robot_motion_model)
        from . import trajectory_computation
        setattr(MobileManipulatorPointMass, 'calculate_trajectory', trajectory_computation.calculate_trajectory)
        setattr(MobileManipulatorPointMass, 'calculate_trajectory_simple', trajectory_computation.calculate_trajectory_simple)
        from . import generate_balls
        setattr(MobileManipulatorPointMass, 'generate_balls', generate_balls.generate_balls)
        setattr(MobileManipulatorPointMass, 'generate_balls_constraints', generate_balls.generate_balls_constraints)
        from . import obstacle_detection_function 
        setattr(MobileManipulatorPointMass, 'object_detection_func', obstacle_detection_function.object_detection_func)

        # init of the parent class
        super().__init__(config)
        self.casadi_model_interface = CasadiModelInterface(config)

    def compute_jacobian_whole(self, current_pose):
        J = self.jacSymMdls['gripped_object'](current_pose)
        return J

    def process_results(self, dict_res, shortest):
        results = []
        times = []
        results.append(dict_res[shortest[0]][shortest[1]])
        times.append(results[0][0][6])
        for i in range(1, len(shortest)-1):
            results.append(dict_res[shortest[i]][shortest[i+1]])
            times.append(results[i][0][6] + times[i-1])
        return results, times

    def forward_kinematics_model(self):
        forward_kinematic = self.kinSymMdls['gripped_object']

        return forward_kinematic
    
    def end_effector_pose_func(self):
        ''' Create a function that computes the end effector pose. '''
        model = self.forward_kinematics_model()

        q = ca.MX.sym('q', self.DoF)
        
        return ca.Function('end_effector_pose', [q], [model(q)[0]])
    
    def end_effector_pose(self, q):
        ''' Compute the end effector pose given the parameters. '''
        return self.end_effector_pose_func()(q)
    
    def base_xyz(self, q):
        ''' Compute the base pose given the parameters. '''
        forward_kinematic = self.kinSymMdls['base']
        return forward_kinematic(q)[0]
    
    def base_jacobian(self, q):
        ''' Compute the base jacobian given the parameters. '''
        J = self.jacSymMdls['base'](q)
        return J





if __name__ == '__main__':
    MobileManipulatorPointMass().plot_dynamic(0.5, np.pi/4, np.pi/6)