from scipy.optimize import fsolve
import numpy as np
import warnings


def calculate_tf(x0, v0, xf, vf, a_max, a_min, v_max, v_min):
    ''' Given the initial and final conditions, calculate the minimum time of flight for the trajectory as well as the parameters of the trajectory.'''
    results = []
    a0 = x0
    a1 = v0
    # print(x0)
    for i in range(2):
        if i == 0:
            a2 = a_max / 2
            a5 = a_min / 2
            guess = v_max
        else:
            a2 = a_min / 2
            a5 = a_max / 2
            guess = v_min
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                a4 = np.sqrt(((a5*(a1**2)) - 4*a2*a5*a0 + 4*a5*a2*xf - a2*(vf**2)) / (a5 - a2))
            except RuntimeWarning as warning:
                # skip to next iteration
                continue
        # a4_attempt or - a4 attempt based on which satisfies the constraints
        if i==1 and -a4 != min(-a4, a1, vf): 
            continue
        if i==0 and a4 != max(a4, a1, vf):
            continue
        if i==1:
            a4 = -a4 # choose the negative root
                     
        ts = (a4 - a1) / (2*a2)
        a3 = a0 + a1*ts + a2*ts**2
        ts_1 = ts
        tf = ((vf - a4) / (2*a5)) + ts
        if ts<0 or tf<=0:
            continue

        if a4 > v_max:
            # v max exceeded
            # Calculate tf assuming v_max is exceeded
            a4 = v_max       

            ts = (a4 - a1) / (2*a2)
            
            a3 = xf - ((a4*(vf - a4))/(2*a5)) - (((vf - a4)**2)/(4*a5))
            ts_1 = ((a3 - a0 - (a1*ts) - (a2*(ts**2)))/a4) + ts

            tf = ((vf - a4) / (2*a5)) + ts_1
        elif a4 < v_min:
            # v min exceeded
            # Calculate tf assuming v_min is exceeded
            ts = (v_min - a1) / (2*a2)
            a4 = v_min

            a3 = xf - ((a4*(vf - a4))/(2*a5)) - (((vf - a4)**2)/(4*a5))
            ts_1 = ((a3 - a0 - (a1*ts) - (a2*(ts**2)))/a4) + ts

            tf = ((vf - a4) / (2*a5)) + ts_1

        if tf > 0 and ts >= 0 and ts_1 >= 0:
            results.append((x0, v0, a2, a3, a4, a5, tf, ts, ts_1))
    if len(results) == 0:
        return None
    # print(results)
    return results


def compute_alpha(x0, v0, xf, vf, a_max, a_min, v_max, v_min, tf):
    ''' Given initial and final states and a desired time of flight, compute the trajectory parameters that satisfy the constraints and the scaling factor alpha (0,1] that scales down the maximum acceleration to meet the constraints.'''
    # get the previous results
    results = []
    a0 = x0
    a1 = v0

    for i in range(2):
        if i == 0:
            a2 = a_max /2
            a5 = a_min / 2
        else:
            a2 = a_min / 2
            a5 = a_max / 2

        # initial_guess = [0.5]
        initial_guess = [0.5, tf/2]
        coefficients = (a0, a1, a2, a5, xf, vf, tf)

        def equations(variables, *args):
            alpha, ts = variables
            a0, a1, a2, a5, xf, vf, tf = args

            eq0 = a0 - alpha*a2*(ts**2) + a1*tf + 2*alpha*tf*ts*(a2 - a5) + alpha*a5*(tf**2)  + alpha*a5*(ts**2) - xf
            eq1 = a1 + (2*alpha*ts*(a2 - a5)) + (2*alpha*a5*tf) - vf

            return [eq0, eq1]

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                solution = fsolve(equations, initial_guess, args=coefficients)
            except RuntimeWarning as warning:
                # skip to next iteration
                continue

        alpha, ts = solution
        # print("initial guess", initial_guess)
        # print("cooefficients", coefficients)
        # print(alpha, ts)
        a4 = a1 + 2*alpha*a2*ts
        a3 = a0 + (a1*ts) + alpha*a2*(ts**2)
        ts_1 = ts
        if ts < 0 or ts >= tf:
            continue

        if a4 > v_max and alpha > 0 and alpha <= 1:
            # v max exceeded
            # Calculate tf assuming v_max is exceeded
            a4 = v_max
            if (4*a2*a5*(a0 + a4*tf -xf)) == 0:
                print('division by zero')
                continue
            alpha = ((2*a5*(a1**2)) + (2*a5*(a4**2)) - (a2*((vf - a4)**2)) - (a5*((a4 - a1)**2)) - (4*a1*a5*a4)) / (4*a2*a5*(a0 + (a4*tf) - xf))
            ts_1 = tf - ((vf - a4) / (2*a5*alpha))
            ts = (a4 - a1) / (2*a2*alpha)
            a3 = a0 + (a1*ts) + (alpha*a2*(ts**2)) + (a4*(ts_1 - ts))

        elif a4 < v_min and alpha > 0 and alpha <= 1:
            a4 = v_min
            # v min exceeded
            if (4*a2*a5*(a0 + a4*tf -xf)) == 0:
                print('division by zero')
                continue
            alpha = ((2*a5*(a1**2)) + (2*a5*(a4**2)) - (a2*((vf - a4)**2)) - (a5*((a4 - a1)**2)) - (4*a1*a5*a4)) / (4*a2*a5*(a0 + (a4*tf) - xf))

            ts_1 = tf - ((vf - a4) / (2*a5*alpha))
            ts = (a4 - a1) / (2*a2*alpha)
            a3 = a0 + (a1*ts) + (alpha*a2*(ts**2)) + (a4*(ts_1 - ts))

        if tf > 0 and ts >= 0 and ts_1 >= 0 and alpha > 0 and alpha <= 1:
            results.append((x0, v0, alpha*a2, a3, a4, alpha*a5, tf, ts, ts_1))
    if len(results) == 0:
        return None
    return results

def space_curve(t, a0, a1, a2, a3, a4, a5, ts, ts_1):
    ''' Given the parameters of the trajectory, compute the position at time t.'''
    if t <= ts:
        return a0 + a1*t + a2*t**2
    elif t <= ts_1:
        return a0 + a1*ts + a2*ts**2 + a4*(t-ts)
    else:
        return a3 + a4*(t-ts_1) + a5*((t-ts_1)**2)

def velocity_curve(t, a1, a2, a3, a4, a5, ts, ts_1):
    ''' Given the parameters of the trajectory, compute the velocity at time t.'''
    if t <= ts:
        return a1 + 2*a2*t
    elif t <= ts_1:
        return a4
    else:
        return a4 + 2*a5*(t-ts_1)

def acceleration_curve(t, a2, a5, ts, ts_1):
    ''' Given the parameters of the trajectory, compute the acceleration at time t.'''
    if t <= ts:
        return 2*a2
    elif t <= ts_1:
        return 0
    else:
        return 2*a5