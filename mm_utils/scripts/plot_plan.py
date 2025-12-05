import matplotlib.pyplot as plt
import numpy as np

from mm_plan.BasePlanner import BasePoseTrajectoryLine
from mm_utils.trajectory_generation import interpolate


def visualize_trajectory(plan):
    n = plan["p"].shape[1]

    f, axes_p = plt.subplots(n, 1, sharex=True)
    if n == 1:
        axes_p = [axes_p]
    for i in range(n):
        axes_p[i].plot(plan["t"], plan["p"][:, i], label="r_" + str(i))
    axes_p[0].set_title("position trajectory")

    f, axes_v = plt.subplots(n, 1, sharex=True)
    if n == 1:
        axes_v = [axes_v]

    for i in range(n):
        axes_v[i].plot(plan["t"], plan["v"][:, i], label="r_" + str(i))
    axes_v[0].set_title("velocity trajectory")

    return [axes_p, axes_v]


def test_interpolation(plan):
    ts = np.random.randint(-1, 20, 10) + np.random.randn(10)

    ps = []
    vs = []

    for t in ts:
        p, v = interpolate(t, plan)
        ps.append(p)
        vs.append(v)

    ps = np.array(ps)
    vs = np.array(vs)

    axes_p, axes_v = visualize_trajectory(plan)
    for i in range(ps.shape[1]):
        axes_p[i].plot(ts, ps[:, i], ".", markersize=10)
        axes_v[i].plot(ts, vs[:, i], ".", markersize=10)


def test_base_pose_trajectory():
    config = BasePoseTrajectoryLine.getDefaultParams()
    config["initial_pose"] = [0, 0, 0.5]
    config["target_pose"] = [2.5, 0, 0]
    config["cruise_speed"] = 0.5
    config["yaw_speed"] = 0.05

    planner = BasePoseTrajectoryLine(config)

    visualize_trajectory(planner.plan)


if __name__ == "__main__":
    test_base_pose_trajectory()
    plt.show()
