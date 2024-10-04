from liecasadi import SE3, SO3, SE3Tangent, SO3Tangent
import numpy as np
import casadi as cs
from spatialmath.base import rotz
from spatialmath import SO3 as SO3s
import matplotlib.pyplot as plt

from mmseq_utils.math import casadi_SO2, casadi_SO3_log, casadi_SO3_Rx

def option_1(x, r):
    R = SO3.from_euler(cs.vcat((cs.MX.zeros(2), x)))
    Rd = SO3.from_euler(cs.vcat((cs.MX.zeros(2), r)))

    e = R-Rd
    J = cs.norm_2(e.vec)
    dJdx = cs.jacobian(J, x)

    e_fcn = cs.Function('J', [x, r], [e.vec])
    J_fcn = cs.Function('J', [x, r], [J])
    dJdx_fcn = cs.Function('J', [x, r], [dJdx])

    return e_fcn, J_fcn, dJdx_fcn

def option_2(x, r):
    Rinv = casadi_SO2(-x)
    Rd = casadi_SO2(r)
    Rerr = Rinv @ Rd
    
    e = cs.atan2(Rerr[1,0], Rerr[0, 0])
    J = e*e
    dJdx = cs.jacobian(J, x)

    e_fcn = cs.Function('e', [x, r], [e])
    J_fcn = cs.Function('J', [x, r], [J])
    dJdx_fcn = cs.Function('dJdx', [x, r], [dJdx])

    return e_fcn, J_fcn, dJdx_fcn

def option_3(x, r):
    R = SO3.from_euler(cs.vcat((cs.MX.zeros(2), x)))
    Rd = SO3.from_euler(cs.vcat((cs.MX.zeros(2), r)))

    v = R.log()
    vd = Rd.log()

    e = vd.vec - v.vec
    J = e.T@e
    e = e[2]
    dJdx = cs.jacobian(J, x)

    e_fcn = cs.Function('e', [x, r], [e])
    J_fcn = cs.Function('J', [x, r], [J])
    dJdx_fcn = cs.Function('dJdx', [x, r], [dJdx])

    return e_fcn, J_fcn, dJdx_fcn

def option_4(x, r):
    Rinv = casadi_SO3_Rx(-x)
    Rd = casadi_SO3_Rx(r)
    Rerr = Rinv @ Rd
    
    e = casadi_SO3_log(Rerr)
    J = e.T@e
    e = e[2]
    dJdx = cs.jacobian(J, x)

    e_fcn = cs.Function('e', [x, r], [e])
    J_fcn = cs.Function('J', [x, r], [J])
    dJdx_fcn = cs.Function('dJdx', [x, r], [dJdx])

    return e_fcn, J_fcn, dJdx_fcn

def plot_error(r, e_fcn, J_fcn, dJdx_fcn):
    xs = np.linspace(-2*np.pi, 2*np.pi, 100)
    rs = np.ones(100) * r

    es = e_fcn(xs, rs)
    Js = J_fcn(xs, rs)
    dJdxs = dJdx_fcn(xs, rs)

    plt.figure()
    plt.plot(xs, es, label="error")
    plt.plot(xs, Js, label="cost")
    plt.plot(xs, dJdxs, label="gradient")
    plt.plot(xs, rs, label='ref')

    plt.xlabel('x')

    plt.legend()

# Random quaternion + normalization
x = cs.MX.sym('x', 1)
r = cs.MX.sym('r', 1)

e_fcn, J_fcn, dJdx_fcn = option_1(x, r)
e_fcn_2, J_fcn_2, dJdx_fcn_2 = option_2(x, r)
e_fcn_3, J_fcn_3, dJdx_fcn_3 = option_3(x, r)
e_fcn_4, J_fcn_4, dJdx_fcn_4 = option_4(x, r)




xnum = np.pi+0.01
rnum = np.pi -0.01
print("x:{}, r:{}, e:{}, dJdx:{}".format(xnum, rnum, e_fcn(xnum, rnum), dJdx_fcn(xnum, rnum)))
print("x:{}, r:{}, e_2:{}, dJdx_2:{}".format(xnum, rnum, e_fcn_2(xnum, rnum), dJdx_fcn_2(xnum, rnum)))
print("x:{}, r:{}, e_3:{}, dJdx_3:{}".format(xnum, rnum, e_fcn_3(xnum, rnum), dJdx_fcn_3(xnum, rnum)))
print("x:{}, r:{}, e_4:{}, dJdx_4:{}".format(xnum, rnum, e_fcn_4(xnum, rnum), dJdx_fcn_4(xnum, rnum)))

R = rotz(rnum)
Rd = rotz(xnum)
Rerr = np.linalg.inv(Rd) @ R
Rerr = rotz(-xnum) @ R
print(rotz(-xnum))
print(rotz(xnum))
print(R)


print(casadi_SO3_log(Rerr))
print(Rerr)
print(np.arctan2(Rerr[1,0],Rerr[0,0]))

plot_error(np.pi-0.01, e_fcn_2, J_fcn_2, dJdx_fcn_2)
plot_error(np.pi-0.01, e_fcn_4, J_fcn_4, dJdx_fcn_4)

plt.show()