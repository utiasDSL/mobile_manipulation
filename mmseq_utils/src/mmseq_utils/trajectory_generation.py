import numpy as np
import matplotlib.pyplot as plt

def sqaure_wave(peak, valley, period, round, dt):
    """ Trajectory generation for sqaure wave

    :param peak: 1D array, peak value
    :param valley: 1Darray, valley value
    :param period: scalar, time it take to go through one peak and one valley
    :param round: integer, total number of period in the trajectory
    :param dt: scalar, time discretization
    :return: a trajectory plan with position and velocity
    """

    n = len(peak)
    if len(valley) != n:
        raise ValueError("peak dimension () should match valley()".format(len(peak), len(valley)))

    N = int(period/dt)
    plan_pos = np.zeros((N, n))
    plan_pos[:int(N/2), :] = peak
    plan_pos[int(N/2):, :] = valley

    plan_pos = np.tile(plan_pos, [round, 1])
    plan_vel = np.zeros_like(plan_pos)
    t = np.arange(N * round) * dt

    return {"t": t, "p": plan_pos, "v": plan_vel}

def interpolate(t, plan):

    if t >= plan["t"][-1]:
        return plan['p'][-1], np.zeros_like(plan['p'][-1])
    elif t <= plan["t"][0]:
        return plan['p'][0], np.zeros_like(plan['p'][0])

    indx = np.argwhere(plan["t"] < t)[-1][0]
    dt = plan["t"][indx + 1] - plan["t"][indx]

    p0 = plan["p"][indx]
    p1 = plan["p"][indx + 1]
    p = (p1 - p0) / dt * (t - indx * dt) + p0

    v0 = plan["v"][indx]
    v1 = plan["v"][indx + 1]
    v = (v1 - v0) / dt * (t - indx * dt) + v0

    return p, v




