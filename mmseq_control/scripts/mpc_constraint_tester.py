import rospy
import rosbag
import numpy as np
import casadi as cs
import os
import matplotlib.pyplot as plt

from mmseq_control.MPCConstraints import SignedDistanceCollisionConstraint
from mmseq_control.map import SDF2D
from mobile_manipulation_central.ros_interface import MapInterface
from cbf_mpc.barrier_function2 import CBF, CBFJacobian


def testSDFMapConstraint(config):
    with rosbag.Bag("/home/tracy/PhD/data/map/cube_map.bag", 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=["/pocd_slam_node/occupied_ef_dist_nodes"]):
                # Assuming at least one marker is present
                if msg.markers:
                    tsdf = msg.markers[0].points
                    tsdf_vals = msg.markers[0].colors

    from mmseq_control.robot import  CasadiModelInterface
    from mmseq_control.MPCCostFunctions import SoftConstraintsRBFCostFunction
    casadi_model_interface = CasadiModelInterface(config["controller"])
    casadi_model_interface.sdf_map.update_map(tsdf, tsdf_vals)

    # query points inside the tsdf points
    pts = np.around(np.array([np.array([p.x,p.y]) for p in tsdf]), 2).reshape((len(tsdf),2))
    xs = pts[:,0]
    ys = pts[:,1]
    x_lim = [min(xs), max(xs)]
    y_lim = [min(ys), max(ys)]
    # Plot sdf map
    casadi_model_interface.sdf_map.vis(x_lim=x_lim,
                                       y_lim=y_lim,
                                       block=False)

    qbs_x = np.linspace(x_lim[0], x_lim[1], int(1.0/0.1 * (x_lim[1] - x_lim[0]))+1)
    qbs_y = np.linspace(y_lim[0], y_lim[1], int(1.0/0.1 * (y_lim[1] - y_lim[0]))+1)
    X,Y= np.meshgrid(qbs_x, qbs_y)
    qbs = np.vstack((X.flatten(), Y.flatten())).T

    robot_mdl = casadi_model_interface.robot
    N = qbs.shape[0] - 1
    nx = robot_mdl.ssSymMdl["nx"]
    nu = robot_mdl.ssSymMdl["nu"]
    x_bar = np.ones((N + 1, nx)) * 0.0
    u_bar = np.ones((N, nu)) * 0
    x_bar[:, :2] = qbs
    d_safe = 0.15
    const = SignedDistanceCollisionConstraint(robot_mdl, casadi_model_interface.getSignedDistanceSymMdls("sdf"), 0.1, N, d_safe, "sdf_base")

    _, g_sdf = const.check(x_bar, u_bar)
    g_sdf = g_sdf.reshape(X.shape)

    # Plot collision constraint
    fig_g, ax_g = plt.subplots()
    levels = np.linspace(-2., 0.5, int(2.5/0.25)+1)
    cs = ax_g.contour(X,Y,g_sdf, levels)
    ax_g.clabel(cs, levels)
    ax_g.grid()
    ax_g.set_title("Collision Constraint $g = -(sd(x) - d_{safe})$, " + "$d_{safe} = $" + f"{0.6 + d_safe}m")   # 0.6 is base collision radius
    ax_g.set_xlabel("x(m)")
    ax_g.set_ylabel("y(m)")
    plt.show(block=False)

    # Plot RBF

    mu = config["controller"]["collision_soft"]['sdf']["mu"]
    zeta = config["controller"]["collision_soft"]['sdf']["zeta"]
    const_soft = SoftConstraintsRBFCostFunction(mu, zeta, const, "SelfCollisionSoftConstraint",expand=False)
    # J_soft = [const_soft.evaluate(x_bar[i,:], u_bar)/X.size for i in range(N+1)]
    J_soft = const_soft.evaluate_vec(x_bar, u_bar)
    J_soft = np.array(J_soft).reshape(X.shape)

    fig_J, ax_J = plt.subplots()
    levels = np.linspace(-0.5, 5, int(5.5/0.5)+1)
    cs = ax_J.contour(X,Y,J_soft, levels)
    ax_J.clabel(cs, levels)
    ax_J.grid()
    ax_J.set_title("Soft Collision Constraint Cost $\mathcal{J} = RBF(g)(x))$, " + "$d_{safe} = $" + f"{0.6 + d_safe}m")   # 0.6 is base collision radius
    ax_J.set_xlabel("x(m)")
    ax_J.set_ylabel("y(m)")
    plt.show()

if __name__ == "__main__":
    # robot mdl
    from mmseq_utils import parsing
    config = parsing.load_config("/home/tracy/Projects/mm_slam/mm_ws/src/mm_sequential_tasks/mmseq_run/config/simple_experiment.yaml")
    testSDFMapConstraint(config)