import numpy as np
import matplotlib.pyplot as plt

def decompose_X(X, state_dim, total_dim):
    x_new = X.full().flatten()
    qs = []
    q_dots = []
    us = []
    for i in range(state_dim):
        qs.append(x_new[i+1::total_dim])
        q_dots.append(x_new[i+1+state_dim::total_dim])
        us.append(x_new[i+1+2*state_dim::total_dim])
    return qs, q_dots, us

def compare_trajectories_casadi_plot(casadi_results, points, dict_res, shortest, forward_kinematic, q_size=2, state_dim=[], labels=[], obstacles=[]):
    # plot X
    number_of_plots = 1 + 2*q_size
    # define the subplots
    rows = int(np.ceil(np.sqrt(number_of_plots)))
    cols = int(np.ceil(number_of_plots / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = np.ravel(axes)

    for i in range(len(casadi_results)):
        tf = float(casadi_results[i][0])
        qs, qs_dots, us = decompose_X(casadi_results[i], q_size, state_dim[i])

        if labels:
            cur_label = labels[i]
        else:
            cur_label = f'{i}'
        # generate trajectory graph
        qs = [i for i in zip(*qs)]
        ee_list = []
        for i in range(len(qs)):
            ee_list.append(forward_kinematic(qs[i])[0])

        x = [float(point[0]) for point in ee_list]
        y = [float(point[1]) for point in ee_list]
        axes[0].plot(x, y, label=f'{cur_label} trajectory')

        t = np.linspace(0, tf, len(x))
        # plot velocities
        for j in range(q_size):
            v = [float(qs_dots[j][k]) for k in range(len(qs_dots[j]))]
            axes[j+1].plot(t, v, label=f'{cur_label} q{j} velocity')
        # plot accelerations
        for j in range(q_size):
            a = [float(us[j][k]) for k in range(len(us[j]))]
            axes[j+q_size+1].plot(t, a, label=f'{cur_label} q{j} acceleration')
    # plot obstacles
    for obstacle in obstacles:
        circle = plt.Circle((obstacle.x, obstacle.y), obstacle.radius, color='grey', alpha=0.5)
        axes[0].add_artist(circle)
    # plot fastest trajectory
    if dict_res!=None:
        tf = 0
        qs = []
        q_dots = []
        us = []
        times = []
        for i in range(len(shortest)-1):
            params = dict_res[shortest[i]][shortest[i+1]]
            t_values = np.linspace(0, params[0][6], 1000)
            for j in range(q_size):
                par = params[j]
                qs_section = [space_curve(t, par[0], par[1], par[2], par[3], par[4], par[5], par[7], par[8]) for t in t_values]
                q_dots_section = [velocity_curve(t, par[1], par[2], par[3], par[4], par[5], par[7], par[8]) for t in t_values]
                us_section = [acceleration_curve(t, par[2], par[5], par[7], par[8]) for t in t_values]
                if i == 0:
                    qs.append(qs_section)
                    q_dots.append(q_dots_section)
                    us.append(us_section)

                else:
                    qs[j]+=qs_section
                    q_dots[j]+=q_dots_section
                    us[j]+=us_section
            times += [t + tf for t in t_values]
            tf += params[0][6]
        #convert qs into ee
        qs = [i for i in zip(*qs)]
        ee_list = []
        for i in range(len(qs)):
            ee_list.append(forward_kinematic(qs[i])[0])
        xs = [float(point[0]) for point in ee_list]
        ys = [float(point[1]) for point in ee_list]
        axes[0].plot(xs, ys, label='Fastest trajectory, sampling algo', color='red')
        # add title
        axes[0].set_title('Trajectories')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        # plot points
        axes[0].scatter([point[0] for point in points], [point[1] for point in points], color='green', label='Points to visit')
        axes[0].legend()

        # plot velocities
        for j in range(q_size):
            v = [float(q_dots[j][k]) for k in range(len(q_dots[j]))]
            axes[j+1].plot(times, v, label=f'Fastest q{j} velocity', color='red')
            axes[j+1].set_title(f'Velocities of q{j}')
            axes[j+1].set_xlabel('Time')
            axes[j+1].set_ylabel('Velocity')
            axes[j+1].legend()
        # plot accelerations
        for j in range(q_size):
            a = [float(us[j][k]) for k in range(len(us[j]))]
            axes[j+q_size+1].plot(times, a, label=f'Fastest q{j} acceleration', color='red')
            axes[j+q_size+1].set_title(f'Accelerations of q{j}')
            axes[j+q_size+1].set_xlabel('Time')
            axes[j+q_size+1].set_ylabel('Acceleration')
            axes[j+q_size+1].legend()
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    plt.show()