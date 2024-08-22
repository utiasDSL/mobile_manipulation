import casadi as ca
import numpy as np

def point_mass_motion_model(state_dim=2):
    x_dyn = ca.MX.sym('x_dyn', 2*state_dim) # assume x_dyn looks like [x0, x1, v0, v1], column vector
    control = ca.MX.sym('control', state_dim) # assume control looks like [u0, u1], column vector
    n_zeros = ca.DM.zeros(state_dim, state_dim)
    n_ones = ca.DM.eye(state_dim)
    matrix = ca.vertcat(
        ca.horzcat(n_zeros, n_ones),
        ca.DM.zeros(state_dim, 2*state_dim)
    )
    rhs_motion = matrix @ x_dyn + ca.vertcat(ca.DM.zeros(state_dim), control)
    motion_model = ca.Function('motion_model', [x_dyn, control], [rhs_motion])
    return motion_model


def forward_kinematic_point_mass(state_dim=2):
    x_dyn = ca.MX.sym('x_dyn', state_dim) # assume x_dyn looks like [x0, x1, v0, v1], column vector

    return ca.Function('forward_kinematic', [x_dyn], [x_dyn])