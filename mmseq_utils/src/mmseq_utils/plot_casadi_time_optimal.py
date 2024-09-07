import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

from mmseq_utils.point_mass_computation_scripts.point_mass_trajectory_optimization import space_curve, velocity_curve, acceleration_curve

def decompose_X(X, state_dim, total_dim):
    x_new = X.full().flatten()
    qs = []
    q_dots = []
    us = []
    for i in range(state_dim):
        qs.append(x_new[i+1::total_dim])
        q_dots.append(x_new[i+1+state_dim::total_dim])
        us.append(x_new[i+1+2*state_dim::total_dim])
    return np.array(qs), np.array(q_dots), np.array(us)

def compare_trajectories_casadi_plot(casadi_results, points, dict_res, shortest, forward_kinematic, state_dim, ts, Ns, q_size=2, labels=[], obstacles=[], show=True, v_bounds=None, a_bounds=None):
    # state dim, ts and Ns are lists

    # plot X
    number_of_plots = 1 + 2*q_size
    # define the subplots
    rows = int(np.ceil(np.sqrt(number_of_plots)))
    cols = int(np.ceil(number_of_plots / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = np.ravel(axes)
    ts = [i.full().flatten() for i in ts]


    for i in range(len(casadi_results)):
        tf = float(casadi_results[i][0])
        qs, qs_dots, us = decompose_X(casadi_results[i], q_size, state_dim[i])

        if labels:
            cur_label = labels[i]
        else:
            cur_label = f'{i}'
        # generate trajectory graph
        qs = [k for k in zip(*qs)]
        ee_list = []
        for j in range(len(qs)):
            ee_list.append(forward_kinematic(qs[j]))

        x = [float(point[0]) for point in ee_list]
        y = [float(point[1]) for point in ee_list]
        axes[0].plot(x, y, label=f'{cur_label}')
        axes[0].set_title('x - y trajectories')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        offset = 0
        if len(ts[i]) == 1:
            t = np.linspace(0, tf, len(x))
        else:
            t = []
            for j in range(len(ts[i])):
                if j > 0:
                    offset += ts[i][j-1] +(ts[i][j]/Ns[i][j])
                t.extend(np.linspace(0, float(ts[i][j]), Ns[i][j])+offset)
        # plot velocities
        for j in range(q_size):
            v = [float(qs_dots[j][k]) for k in range(len(qs_dots[j]))]
            axes[j+1].plot(t, v, label=f'{cur_label}')
            if v_bounds:
                axes[j+1].plot(t, [v_bounds[0][j]]*len(t), 'r--')
                axes[j+1].plot(t, [v_bounds[1][j]]*len(t), 'r--')
            axes[j+1].set_title(f'Velocities of q{j}')
            axes[j+1].set_xlabel('Time')
            axes[j+1].set_ylabel('Velocity')
        # plot accelerations
        for j in range(q_size):
            a = [float(us[j][k]) for k in range(len(us[j]))]
            if a_bounds:
                axes[j+q_size+1].plot(t, [a_bounds[i][0][j]]*len(t), 'r--')
                axes[j+q_size+1].plot(t, [a_bounds[i][1][j]]*len(t), 'r--')
            axes[j+q_size+1].plot(t, a, label=f'{cur_label}')
            axes[j+q_size+1].set_title(f'Accelerations of q{j}')
            axes[j+q_size+1].set_xlabel('Time')
            axes[j+q_size+1].set_ylabel('Acceleration')
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
            ee_list.append(forward_kinematic(qs[i]))
        xs = [float(point[0]) for point in ee_list]
        ys = [float(point[1]) for point in ee_list]
        axes[0].plot(xs, ys, label='Sampling algo', color='red')
       
        # plot velocities
        for j in range(q_size):
            v = [float(q_dots[j][k]) for k in range(len(q_dots[j]))]
            
            # axes[j+1].legend()
        # plot accelerations
        for j in range(q_size):
            a = [float(us[j][k]) for k in range(len(us[j]))]
            axes[j+q_size+1].plot(times, a, label=f'Sampled', color='red')
            
            # axes[j+q_size+1].legend()
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
    # plot points
    axes[0].scatter([point[0] for point in points[1:]], [point[1] for point in points[1:]], color='green', label='Points to visit')
    # plot start point
    axes[0].scatter(points[0][0], points[0][1], color='purple', label='Start point')

    axes[0].legend()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    if not show:
        return fig
    plt.show()


def plot_3d_trajectory(casadi_results, points, dict_res, shortest, forward_kinematic, state_dim, ts, Ns, q_size=2, labels=[], obstacles=[], title='Trajectories', show=True):
    fig = plt.figure(figsize=(15, 10))  
    ax = fig.add_subplot(111, projection='3d')
    ts = [i.full().flatten() for i in ts]
    for i in range(len(casadi_results)):
        tf = float(casadi_results[i][0])
        qs, qs_dots, us = decompose_X(casadi_results[i], q_size, state_dim[i])

        if labels:
            cur_label = labels[i]
        else:
            cur_label = f'{i}'
        # generate trajectory graph
        qs = [k for k in zip(*qs)]
        ee_list = []
        for j in range(len(qs)):
            ee_list.append(forward_kinematic(qs[j]))

        x = [float(point[0]) for point in ee_list]
        y = [float(point[1]) for point in ee_list]
        z = [float(point[2]) for point in ee_list]
        ax.plot(x, y, z, label=f'{cur_label} trajectory')

        offset = 0
        if len(ts[i]) == 1:
            t = np.linspace(0, tf, len(x))
        else:
            t = []
            for j in range(len(ts[i])):
                if j > 0:
                    offset += ts[i][j-1] + (ts[i][j]/Ns[i][j])
                t.extend(np.linspace(0, float(ts[i][j]), Ns[i][j])+offset)
    # plot fastest trajectory
    if dict_res!=None:
        tf = 0
        qs = []
        times = []
        for i in range(len(shortest)-1):
            params = dict_res[shortest[i]][shortest[i+1]]
            t_values = np.linspace(0, params[0][6], 1000)
            for j in range(q_size):
                par = params[j]
                qs_section = [space_curve(t, par[0], par[1], par[2], par[3], par[4], par[5], par[7], par[8]) for t in t_values]
                if i == 0:
                    qs.append(qs_section)
                else:
                    qs[j]+=qs_section
            times += [t + tf for t in t_values]
            tf += params[0][6]
        #convert qs into ee
        qs = [i for i in zip(*qs)]
        ee_list = []
        for i in range(len(qs)):
            ee_list.append(forward_kinematic(qs[i]))
        xs = [float(point[0]) for point in ee_list]
        ys = [float(point[1]) for point in ee_list]
        zs = [float(point[2]) for point in ee_list]
        ax.plot(xs, ys, zs, label='Fastest trajectory, sampling algo', color='red')
    # plot points
    ax.scatter([point[0] for point in points[1:]], [point[1] for point in points[1:]], [point[2] for point in points[1:]], color='green', label='Points to visit')
    # plot start point
    ax.scatter(points[0][0], points[0][1], points[0][2], color='purple', label='Start point')
    ax.legend()
    # add title
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if not show:
        return fig
    plt.show()

def plot_whisker_plots(data, x_axis, show=True, titles=[], ylabel='Values'):
    fig, ax = plt.subplots(1, len(data), figsize=(15, 10))
    if len(data) == 1:
        ax = [ax]
    for i in range(len(data)):
        ax[i].boxplot(data[i])
        if titles:
            title = titles[i]
        else:
            title = 'Whisker plot'
        ax[i].set_title(f'{title}')
        ax[i].set_xticklabels(x_axis, rotation=20)
        ax[i].set_ylabel(ylabel)
        ax[i].grid()
        # overaly the points
        for j in range(len(data[i])):
            y = data[i][j]
            x = np.ones_like(y)*(j+1)
            ax[i].plot(x, y, 'b.')
    if not show:
        return fig
    plt.show()

def plot_obstacle_avoidance(casadi_results, obstacles_avoidance, state_dim, q_size=2, labels=[], show=True, limit=0):
    number_of_plots = max(obstacles_avoidance(np.zeros(q_size)).shape[0], obstacles_avoidance(np.zeros(q_size)).shape[1])
    # define the subplots
    rows = int(np.ceil(np.sqrt(number_of_plots)))
    cols = int(np.ceil(number_of_plots / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = np.ravel(axes)
    obstacles_violated = False
    for i in range(len(casadi_results)):
        qs, qs_dots, us = decompose_X(casadi_results[i], q_size, state_dim[i])
        # generate trajectory graph
        qs = [k for k in zip(*qs)]
        obstacle_values = []
        for j in range(len(qs)):
            temp = obstacles_avoidance(qs[j]).full().flatten()
            obstacle_values.append(temp)
            # check all values are >0
            if np.any(temp < 0):
                print(f'Obstacle avoidance violated for pair {i}')
                print(f'with minimum value {np.min(temp)}')
                obstacles_violated = True

        obstacle_values = np.array(obstacle_values).T
        x_axis = np.arange(0, len(qs))
        for j in range(number_of_plots):
            axes[j].plot(x_axis, obstacle_values[j], label=f'{labels[i]}')
            # plot bound
            axes[j].plot(x_axis, [limit]*len(qs), 'r--')

    for j in range(number_of_plots):
        axes[j].set_title(f'Obstacle avoidance for pair {j}')
        axes[j].set_xlabel('Node number')
        axes[j].set_ylabel('Signed distance')
        if j == 0:
            axes[j].legend()

    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    if not show:
        return fig, obstacles_violated
    plt.show()

def plot_base_ee_velocities(casadi_results, motion_class, state_dim, ts, Ns, q_size=2, labels=[], show=True):
    # define the subplots
    rows = 3
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = np.ravel(axes)
    velocities_titles = ['Base velocity x', 'Base velocity y', 'End effector velocity x', 'End effector velocity y', 'End effector velocity z']
    ts = [i.full().flatten() for i in ts]

    for i in range(len(casadi_results)):
        tf = float(casadi_results[i][0])
        qs, qs_dots, us = decompose_X(casadi_results[i], q_size, state_dim[i])
        # generate trajectory graph
        qs = [k for k in zip(*qs)]
        qs_dots = [k for k in zip(*qs_dots)]
        velocities = []

        for j in range(len(qs)):
            v_base = motion_class.base_jacobian(qs[j]) @ qs_dots[j]
            v_ee = motion_class.compute_jacobian_whole(qs[j]) @ qs_dots[j]
            velocities.append([*v_base.full().flatten(), *v_ee.full().flatten()])
    
        velocities = np.array(velocities).T

        offset = 0
        if len(ts[i]) == 1:
            t = np.linspace(0, tf, len(qs))
        else:
            t = []
            for j in range(len(ts[i])):
                if j > 0:
                    offset += ts[i][j-1] + (ts[i][j]/Ns[i][j])
                t.extend(np.linspace(0, float(ts[i][j]), Ns[i][j])+offset)
        
        for j in range(5):
            axes[j].plot(t, velocities[j], label=f'{labels[i]}')
            axes[j].set_title(velocities_titles[j])
            axes[j].set_xlabel('Time')
            axes[j].set_ylabel('Velocity')

    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    if not show:
        return fig
    plt.show()

def simple_plot(x, Y, titles=['Simple plot'], x_label='x', y_label='y', show=True, labels=[]):
    fig, ax = plt.subplots(1, len(Y), figsize=(15, 10))
    if len(Y) == 1:
        ax = [ax]
    for i in range(len(Y)):
        for k in range(len(Y[i])):
            y = Y[i][k]
            if labels:
                ax[i].plot(x, y, 'o--', label=f'{labels[k]}')
            else:
                ax[i].plot(x, y, 'o--')
        ax[i].set_title(titles[i])
        ax[i].set_xlabel(x_label)
        ax[i].set_ylabel(y_label)
        if labels:
            ax[i].legend()
    if not show:
        return fig
    plt.show()

def plot_base_ee_pos(casadi_results, motion_class, state_dim, ts, Ns, q_size=2, labels=[], show=True):
    # define the subplots
    rows = 3
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = np.ravel(axes)
    velocities_titles = ['Base traj x', 'Base traj y', 'End effector traj x', 'End effector traj y', 'End effector traj z']
    ts = [i.full().flatten() for i in ts]

    for i in range(len(casadi_results)):
        tf = float(casadi_results[i][0])
        qs, qs_dots, us = decompose_X(casadi_results[i], q_size, state_dim[i])
        # generate trajectory graph
        qs = [k for k in zip(*qs)]
        velocities = []
        ps = []

        for j in range(len(qs)):
            p_base = motion_class.base_xyz(qs[j])
            p_ee = motion_class.end_effector_pose(qs[j])
            ps.append([*p_base.full().flatten(), *p_ee.full().flatten()])

        ps = np.array(ps).T
        print(ps)
        offset = 0
        if len(ts[i]) == 1:
            t = np.linspace(0, tf, len(qs))
        else:
            t = []
            for j in range(len(ts[i])):
                if j > 0:
                    offset += ts[i][j-1] + (ts[i][j]/Ns[i][j])
                t.extend(np.linspace(0, float(ts[i][j]), Ns[i][j])+offset)
        
        for j in range(5):
            axes[j].plot(t, ps[j], label=f'{labels[i]}')
            axes[j].set_title(velocities_titles[j])
            axes[j].set_xlabel('Time')
            axes[j].set_ylabel('Trajectory')

    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    if not show:
        return fig
    plt.show()

def plot_motion_model(casadi_results, motion_model, state_dim, q_size=2, labels=[], show=True, limit=0):
    number_of_plots = q_size
    # define the subplots
    rows = int(np.ceil(np.sqrt(number_of_plots)))
    cols = int(np.ceil(number_of_plots / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = np.ravel(axes)
    
    for i in range(len(casadi_results)):
        qs, qs_dots, us = decompose_X(casadi_results[i], q_size, state_dim[i])
        qs, qs_dots, us = np.array(qs).T, np.array(qs_dots).T, np.array(us).T
        N = len(qs)
        dt = float(casadi_results[i][0])/N
        constraints_values = []

        for j in range(N-1):
        
        # generate trajectory 
            X_j = ca.vertcat(qs[j], qs_dots[j])
            X_j1 = ca.vertcat(qs[j+1], qs_dots[j+1])
            differences = X_j1 - X_j - dt*motion_model(X_j, us[j])
            constraints_values.append(differences.full().flatten())
        x_axis = np.arange(0, N-1)
        constraints_values = np.array(constraints_values).T

        for k in range(q_size):
            axes[k].plot(x_axis, constraints_values[k], label=f'{labels[i]}')
            # plot bound
            axes[k].plot(x_axis, [limit]*(N-1), 'r--', label='')

    for j in range(number_of_plots):
        axes[j].set_title(f'Motion model constraint for joint {j}')
        axes[j].set_xlabel('Node number')
        # axes[j].set_ylabel('Signed distance')
        if j == 0:
            axes[j].legend()

    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    if not show:
        return fig, obstacles_violated
    plt.show()
    
    


