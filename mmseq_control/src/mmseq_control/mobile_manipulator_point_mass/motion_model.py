import casadi as ca

def robot_motion_model(self):
    x = ca.MX.sym('x')
    theta1 = ca.MX.sym('theta1')
    theta2 = ca.MX.sym('theta2')

    x_dot = ca.MX.sym('x_dot')
    t1_dot = ca.MX.sym('t1_dot')
    t2_dot = ca.MX.sym('t2_dot')

    q = ca.vertcat(x, theta1, theta2)
    q_dot = ca.vertcat(x_dot, t1_dot, t2_dot)
    u = ca.MX.sym('u', 3)
    inputs = ca.vertcat(q, q_dot)

    l1 = self.arm_length1
    I2 = (1/3)*self.m2*(self.arm_length1**2)
    I3 = (1/3)*self.m3*(self.arm_length2**2)
    c2 = 0.5*self.m2*self.arm_length1
    c3 = 0.5*self.m3*self.arm_length2
    g = 9.81
    l_c_m_2 = 0.5*self.arm_length1
    l_c_m_3 = 0.5*self.arm_length2

    M = ca.vertcat(
        ca.horzcat(self.m1 + self.m2 + self.m3, -ca.sin(theta1)*c2 - self.m3*l1*ca.sin(theta1) - c3*ca.sin(theta1+theta2), -ca.sin(theta1 + theta2)*c3),
        ca.horzcat(-ca.sin(theta1)*c2 - self.m3*l1*ca.sin(theta1) - c3*ca.sin(theta1+theta2), (l1**2)*self.m3 + 2*c3*ca.cos(theta2)*l1+ I3 + I2, c3*ca.cos(theta2)*l1 + I3),
        ca.horzcat(-ca.sin(theta1 + theta2)*c3, l1*ca.cos(theta2)*c3 + I3, I3)
    )

    fg = - ca.vertcat(0, self.m2*g*ca.cos(theta1)*l_c_m_2 + self.m3*g*(ca.cos(theta1+theta2)*l_c_m_3 + l1*ca.cos(theta1)), self.m3*g*(ca.cos(theta1+theta2)*l_c_m_3))

    M_dot = ca.vertcat(
        ca.horzcat(0, (-t1_dot*c2 - self.m3*t1_dot*l1)*ca.cos(theta2) - (t1_dot + t2_dot)*c3*ca.cos(theta1+theta2), -(t1_dot+t2_dot)*ca.cos(theta1+theta2)*c3),
        ca.horzcat((-t1_dot*c2 - self.m3*t1_dot*l1)*ca.cos(theta2) - (t1_dot + t2_dot)*c3*ca.cos(theta1+theta2), -2*t2_dot*l1*c3*ca.sin(theta2), -c3*l1*t2_dot*ca.sin(theta2)),
        ca.horzcat(-(t1_dot+t2_dot)*ca.cos(theta1+theta2)*c3, -c3*l1*t2_dot*ca.sin(theta2), 0)
    )

    d_M_d_t1 = ca.jacobian(M, theta1).reshape((3, 3))
    d_M_d_t2 = ca.jacobian(M, theta2).reshape((3, 3))


    h = M_dot@q_dot + ca.vertcat(0,
                                q_dot.T @ d_M_d_t1 @ q_dot, 
                                q_dot.T @ d_M_d_t2 @ q_dot)

    n_zeros = ca.DM.zeros(3, 3)
    n_ones = ca.DM.eye(3)
    matrix = ca.vertcat(
        ca.horzcat(n_zeros, n_ones),
        ca.DM.zeros(3, 6)
    )
    rhs_motion = matrix @ inputs + ca.vertcat(ca.DM.zeros(3), ca.inv(M)@(u + fg - h))
    # rhs_motion = matrix @ inputs + ca.vertcat(ca.DM.zeros(3), u)

    motion_model = ca.Function('motion_model',[inputs, u], [rhs_motion])
    # print('done')
    return motion_model
    