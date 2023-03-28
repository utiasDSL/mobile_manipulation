import argparse
import datetime
import matplotlib.pyplot as plt

import numpy as np
import casadi as cs
import casadi_kin_dyn.py3casadi_kin_dyn as cas_kin_dyn
from scipy.linalg import expm

from liegroups import SO3
from mmseq_utils import parsing
from mmseq_simulator import simulation

import yappi
class MobileManipulator3D:

    def __init__(self, config):
        """ Casadi simbolic model of Mobile Manipulator

        """
        urdf_path = parsing.parse_and_compile_urdf(config["robot"]["urdf"])
        # urdf_path = parsing.parse_ros_path(config["robot"]["urdf"])
        urdf = open(urdf_path, 'r').read()
        self.kindyn = cas_kin_dyn.CasadiKinDyn(urdf)  # construct main class

        self.numjoint = self.kindyn.nq()
        self.DoF = self.numjoint + 3

        self.qb_sym = cs.MX.sym('qb', 3)
        self.qa_sym = cs.MX.sym('qa', self.numjoint)
        self.q_sym = cs.vertcat(self.qb_sym, self.qa_sym)

        self.ub_x = parsing.parse_array(config["robot"]["limits"]["state"]["upper"])
        self.lb_x = parsing.parse_array(config["robot"]["limits"]["state"]["lower"])
        self.ub_u = parsing.parse_array(config["robot"]["limits"]["input"]["upper"])
        self.lb_u = parsing.parse_array(config["robot"]["limits"]["input"]["lower"])

        self.link_names = config["robot"]["link_names"]
        self.tool_link_name = config["robot"]["tool_link_name"]
        self._setupKinSymMdl()
        self._setupSSSymMdlDI()
        self._setupJacobianSymMdl()

    def _setupSSSymMdlDI(self):
        """ Create State-space symbolic model for MM

        """
        self.va_sym = cs.MX.sym('va', self.numjoint)
        self.vb_sym = cs.MX.sym('vb', 3)                    # Assuming nonholonomic vehicle, velocity in world frame
        self.v_sym = cs.vertcat(self.vb_sym, self.va_sym)

        self.x_sym = cs.vertcat(self.q_sym, self.v_sym)
        self.u_sym = cs.MX.sym('u', self.v_sym.size()[0])

        nx = self.x_sym.size()[0]
        nu = self.u_sym.size()[0]
        self.ssSymMdl = {"x": self.x_sym,
                         "u": self.u_sym,
                         "mdl_type": ["linear", "time_invariant"],
                         "nx": nx,
                         "nu": nu,
                         "ub_x": list(self.ub_x),
                         "lb_x": list(self.lb_x),
                         "ub_u": list(self.ub_u),
                         "lb_u": list(self.lb_u)}

        A = cs.DM.zeros((nx, nx))
        G = cs.DM.eye(self.DoF)
        A[:self.DoF, self.DoF:] = G
        B = cs.DM.zeros((nx, nu))
        B[self.DoF:, :] = cs.DM.eye(nu)

        xdot = A @ self.x_sym + B @ self.u_sym
        fmdl = cs.Function("ss_fcn", [self.x_sym, self.u_sym], [xdot], ["x", "u"], ["xdot"])
        self.ssSymMdl["fmdl"] = fmdl.expand()
        self.ssSymMdl["A"] = A
        self.ssSymMdl["B"] = B
        self.ssSymMdl["fmdlk"] = self._discretizefmdl(self.ssSymMdl)

    def _setupKinSymMdl(self):
        """ Create kinematic symbolic model for MM links keyed by link name

        """
        self.kinSymMdls = {}
        for name in self.link_names:
            self.kinSymMdls[name] = self._getFk(name)

    def _setupJacobianSymMdl(self):
        self.jacSymMdls = {}
        for name in self.link_names:
            fk_fcn = self.kinSymMdls[name]
            fk_pos_eqn, _ = fk_fcn(self.q_sym)
            Jk_eqn = cs.jacobian(fk_pos_eqn, self.q_sym)
            self.jacSymMdls[name] = cs.Function(name + "_jac_fcn", [self.q_sym], [Jk_eqn], ["q"], ["J(q)"])

    def _getFk(self, link_name):
        """ Create symbolic function for a link named link_name
            The symbolic function returns the position of its parent joint in and rotation w.r.t the world frame.
            Note this is different from link_state provided by Pybullet which provides CoM position.

        """
        if link_name == "base":
            return cs.Function(link_name + "_fcn", [self.q_sym], [self.qb_sym[:2], self.qb_sym[2]], ["q"], ["pos2", "heading"])

        Hwb = cs.MX.eye(4)
        Hwb[0, 0] = np.cos(self.qb_sym[2])
        Hwb[1, 0] = np.sin(self.qb_sym[2])
        Hwb[0, 1] = -np.sin(self.qb_sym[2])
        Hwb[1, 1] = np.cos(self.qb_sym[2])
        Hwb[:2, 3] = self.qb_sym[:2]

        fk_str = self.kindyn.fk(link_name)
        fk = cs.Function.deserialize(fk_str)
        link_pos, link_rot = fk(self.qa_sym)
        Hbl = cs.MX.eye(4)
        Hbl[:3, :3] = link_rot
        Hbl[:3, 3] = link_pos
        Hwl = Hwb @ Hbl

        return cs.Function(link_name + "_fcn", [self.q_sym], [Hwl[:3, 3], Hwl[:3, :3]], ["q"], ["pos", "rot"])

    def _discretizefmdl(self, ss_mdl):
        if "linear" in ss_mdl["mdl_type"]:
            x_sym = ss_mdl["x"]
            u_sym = ss_mdl["u"]
            dt_sym = cs.MX.sym("dt")
            A = ss_mdl["A"]
            B = ss_mdl["B"]
            nx = x_sym.size()[0]
            nu = u_sym.size()[0]

            M = np.zeros((nx + nu, nx + nu))
            M[:nx, :nx] = A
            M[:nx, nx:] = B
            Md = expm(M * 0.1)
            Ad = Md[:nx, :nx]
            Bd = Md[:nx, nx:]

            xk1_eqn = Ad @ x_sym + Bd @ u_sym
            fdsc_fcn = cs.Function("fmdlk", [x_sym, u_sym], [xk1_eqn], ["xk", "uk"], ["xk1"])

        return fdsc_fcn

    @staticmethod
    def ssIntegrate(dt, xo, u_bar, ssSymMdl):
        """

        :param dt: discretization time step
        :param x0: initial state
        :param u_bar: control inputs numpy.ndarray [N, nu]
        :param ssSymMdl: state-space symbolic model
        :return: x_bar
        """
        N = u_bar.shape[0]
        if "linear" in ssSymMdl["mdl_type"]:
            fk = ssSymMdl["fmdlk"]
            f_pred = fk.mapaccum(N)
            x_bar = f_pred(xo, u_bar.T)
            x_bar = np.hstack((np.expand_dims(xo, -1), x_bar)).T

        return x_bar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration file.")
    parser.add_argument(
        "--video",
        nargs="?",
        default=None,
        const="",
        help="Record video. Optionally specify prefix for video directory.",
    )
    args = parser.parse_args()

    # load configuration
    config = parsing.load_config(args.config)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    # Create Sym Mdl
    robot = MobileManipulator3D(ctrl_config)


    # start the simulation
    timestamp = datetime.datetime.now()
    sim = simulation.BulletSimulation(
        config=sim_config, timestamp=timestamp, cli_args=args
    )
    mm = sim.robot
    # mm.command_velocity(np.zeros(9))
    sim.settle(5.0)

    for name in robot.link_names[1:]:
        print(name)
        q, v = mm.joint_states()

        link_idx = mm.links[name][0]
        pos_sim, orn_sim = mm.link_pose(link_idx)
        rot_sim = SO3.from_quaternion(orn_sim, 'xyzw').as_matrix()
        J_sim = mm.jacobian(q)

        fk_fcn = robot.kinSymMdls[name]
        # yappi.set_clock_type("wall")
        # yappi.start()
        pos_mdl, rot_mdl = fk_fcn(q)
        J_fcn = robot.jacSymMdls[name]
        J_mdl = J_fcn(q)
        # print(J_mdl)
        # print(J_sim[:3])
        # yappi.get_func_stats().print_all()

        pos_mdl = pos_mdl.toarray().flatten()
        # Note that position differences won't be zero because pybullet gives CoM position whereas casadi_kin_dyn gives joint position
        # The differences should be exactly the CoM position in world frame
        print("pos diff:{}, rot diff{}".format(np.linalg.norm(pos_sim - pos_mdl), np.linalg.norm(rot_mdl - rot_sim)))
        print("J diff{}".format(np.linalg.norm(J_mdl - J_sim[:3])))

    print("Testing motion model integrator")
    dt = 0.1
    a = 1.
    N = 10
    u_bar = np.array([[a]*9]*N)
    xo = np.array(mm.joint_states()).flatten()
    x_bar_sym = MobileManipulator3D.ssIntegrate(dt, xo, u_bar, robot.ssSymMdl)

    x_bar_num = np.zeros((N+1, 18))
    x_bar_num[0] = xo
    for k in range(N):
        x_bar_num[k+1, 9:] = x_bar_num[k, 9:] + a * dt
        x_bar_num[k+1,:9] = x_bar_num[k, :9] + x_bar_num[k, 9:] * dt + 0.5 * a * dt * dt

    pred_diff = np.linalg.norm(x_bar_num - x_bar_sym)
    print("Prediction diff:{}".format(pred_diff))
    tgrid = np.arange(N + 1) * dt
    plt.figure(1)
    plt.clf()
    plt.plot(tgrid, x_bar_sym[:, 0], 'r--')
    plt.plot(tgrid, x_bar_num[:, 0], 'b-')
    plt.plot(tgrid, x_bar_sym[:, 6], 'r--')
    plt.plot(tgrid, x_bar_num[:, 6], 'b-')
    plt.xlabel('t')
    plt.legend(['x0_sym', 'x0_num', 'x6_sym', 'x6_num'])
    plt.grid()
    plt.show()

    print("finished")
    
    
if __name__ == "__main__":
    main()

            
        
        
        
        
        
        
        
        
        
        