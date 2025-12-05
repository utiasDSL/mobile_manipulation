import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import rospy

from mm_control.MPCConstraints import SignedDistanceConstraint
from mm_control.MPCCostFunctions import SoftConstraintsRBFCostFunction
from mm_utils import parsing


def test_SignedDistanceConstraint_2D(config):
    from mobile_manipulation_central.ros_interface import MapInterfaceNew

    map_ros_interface = MapInterfaceNew(config["controller"])
    rate = rospy.Rate(100)

    while not map_ros_interface.ready() and not rospy.is_shutdown():
        rate.sleep()

    from mm_control.robot import CasadiModelInterface

    casadi_model_interface = CasadiModelInterface(config["controller"])
    while not rospy.is_shutdown():
        _, map = map_ros_interface.get_map()
        casadi_model_interface.sdf_map.update_map(*map)
        tsdf, _tsdf_vals = map_ros_interface.tsdf, map_ros_interface.tsdf_vals
        # query points inside the tsdf points
        pts = np.around(np.array([np.array([p.x, p.y]) for p in tsdf]), 2).reshape(
            (len(tsdf), 2)
        )
        xs = pts[:, 0]
        ys = pts[:, 1]
        x_lim = [min(xs), max(xs)]
        y_lim = [min(ys), max(ys)]

        # Plot sdf map
        casadi_model_interface.sdf_map.vis(x_lim=x_lim, y_lim=y_lim, block=False)

        qbs_x = np.linspace(
            x_lim[0], x_lim[1], int(1.0 / 0.1 * (x_lim[1] - x_lim[0])) + 1
        )
        qbs_y = np.linspace(
            y_lim[0], y_lim[1], int(1.0 / 0.1 * (y_lim[1] - y_lim[0])) + 1
        )
        X, Y = np.meshgrid(qbs_x, qbs_y)

        # Plot collision constraint
        robot_mdl = casadi_model_interface.robot
        N = X.size
        nx = robot_mdl.ssSymMdl["nx"]
        nu = robot_mdl.ssSymMdl["nu"]
        x = np.ones((nx, N)) * 0
        x[0, :] = X.flatten()
        x[1, :] = Y.flatten()
        u = np.ones((nu, N)) * 0
        d_safe = 0.15
        const = SignedDistanceConstraint(
            robot_mdl,
            casadi_model_interface.getSignedDistanceSymMdls("sdf"),
            d_safe,
            "sdf",
        )
        params = casadi_model_interface.sdf_map.get_params()
        param_map = const.p_struct(0)
        param_map["x_grid"] = params[0]
        param_map["y_grid"] = params[1]
        param_map["value"] = params[2]

        g_sdf = const.check(x, u, param_map.cat).toarray()
        g_sdf = g_sdf.flatten().reshape(X.shape)
        print(const.get_p_dict())

        _, ax_g = plt.subplots()
        levels = np.linspace(-2.0, 0.5, int(2.5 / 0.25) + 1)
        cs = ax_g.contour(X, Y, g_sdf, levels)
        ax_g.clabel(cs, levels)
        ax_g.grid()
        ax_g.set_title(
            "Collision Constraint $g = -(sd(x) - d_{safe})$, "
            + "$d_{safe} = $"
            + f"{0.6 + d_safe}m"
        )  # 0.6 is base collision radius
        ax_g.set_xlabel("x(m)")
        ax_g.set_ylabel("y(m)")
        plt.show(block=False)

        # Plot Softened collision function
        mu = config["controller"]["collision_soft"]["sdf"]["mu"]
        zeta = config["controller"]["collision_soft"]["sdf"]["zeta"]
        print(const.get_p_dict())
        const_soft = SoftConstraintsRBFCostFunction(
            mu, zeta, const, "SelfCollisionSoftConstraint", expand=False
        )
        # J_soft = [const_soft.evaluate(x_bar[i,:], u_bar)/X.size for i in range(N+1)]
        J_soft = const_soft.evaluate_vec(x, u, param_map.cat)
        J_soft = np.array(J_soft).reshape(X.shape)

        _, ax_J = plt.subplots()
        levels = np.linspace(-0.5, 5, int(5.5 / 0.5) + 1)
        cs = ax_J.contour(X, Y, J_soft, levels)
        ax_J.clabel(cs, levels)
        ax_J.grid()
        ax_J.set_title(
            "Soft Collision Constraint Cost $\mathcal{J} = RBF(g)(x))$, "
            + "$d_{safe} = $"
            + f"{0.6 + d_safe}m"
        )  # 0.6 is base collision radius
        ax_J.set_xlabel("x(m)")
        ax_J.set_ylabel("y(m)")
        plt.show()


if __name__ == "__main__":
    config_path = parsing.parse_ros_path(
        {"package": "mm_run", "path": "config/simple_experiment.yaml"}
    )
    config = parsing.load_config(config_path)
    rospy.init_node("constraints_tester")

    argv = rospy.myargv(argv=sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sdf2d", action="store_true", help="Test SignedDistanceConstraint"
    )

    args = parser.parse_args(argv[1:])

    if args.sdf2d:
        test_SignedDistanceConstraint_2D(config)
