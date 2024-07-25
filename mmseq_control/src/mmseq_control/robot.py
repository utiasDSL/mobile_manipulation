import argparse
import datetime
import matplotlib.pyplot as plt
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer as visualizer
from typing import List, Dict

import numpy as np
from numpy import ndarray
import casadi as cs
import casadi_kin_dyn.py3casadi_kin_dyn as cas_kin_dyn
import rospkg
from scipy.linalg import expm
import hppfcl as fcl
from spatialmath.base import r2q, rotz, q2r

# from liegroups import SO3
import mmseq_control.map as map
from mmseq_utils import parsing
from mmseq_simulator import simulation
from mobile_manipulation_central.kinematics import RobotKinematics
# from cbf_mpc.barrier_function2 import CBF, CBFJacobian

# import yappi

def signed_distance_sphere_sphere(c1, c2, r1, r2):
    """ signed distance between two spheres

    :param c1: numpy array or casadi sym of size 3, center of sphere 1
    :param c2: numpy array or casadi sym of size 3, center of sphere 2
    :param r1: scalar, radius of sphere 1
    :param r2: scalar, radius of sphere 2
    :return:
    """
    return cs.norm_2(c1 - c2) - r1 -r2

def signed_distance_half_space_sphere(d, p, n, c, r):
    """ signed distance of a sphere to half space

    :param d: scalar, offset from p along n
    :param p: vector, offset of the normal vector
    :param n: numpy array, normal vector of the plane
    :param c: numpy array or casadi sym of size 3, center of the sphere
    :param r: scalar, radius of the sphere
    :return:
    """

    return (c - p).T @ n - d - r

def signed_distance_sphere_cylinder(c_sphere, c_cylinder, r_sphere, r_cylinder):
    """ signed distance between a sphere and a cylinder (with infinite height)

    :param c_sphere: center of the sphere
    :param c_cylinder: center of the cylinder
    :param r_sphere: radius of the sphere
    :param r_cylinder: radius of the cylinder
    :return:
    """


    return cs.norm_2(c_sphere[:2]-c_cylinder[:2]) - r_sphere - r_cylinder


class PinocchioInterface:

    def __init__(self, config):
        # 1. build robot model
        urdf_path = parsing.parse_and_compile_urdf(config["robot"]["urdf"])
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(urdf_path)
        # 2. add scene model
        if config["scene"]["enabled"]:
            scene_urdf_path = parsing.parse_and_compile_urdf(config["scene"]["urdf"])
            # here models are passed in so that scene models can be appended to robot model
            pin.buildModelFromUrdf(scene_urdf_path, self.model)
            pin.buildGeomFromUrdf(self.model, scene_urdf_path, pin.GeometryType.COLLISION, self.collision_model)
            pin.buildGeomFromUrdf(self.model, scene_urdf_path, pin.GeometryType.VISUAL, self.visual_model)
        self.addGroundCollisionObject()

        self.collision_link_names = config["robot"]["collision_link_names"].copy()
        if config["scene"]["enabled"]:
            self.collision_link_names.update(config["scene"]["collision_link_names"])

    def visualize(self, q):
        viz = visualizer(self.model, self.collision_model, self.visual_model)
        viz.initViewer(open=True)
        viz.viewer.open()
        viz.loadViewerModel()
        viz.display(q)

    def getGeometryObject(self, link_names):
        objs = []
        for name in link_names:
            o_id = self.collision_model.getGeometryId(name + "_0")
            if o_id >= self.collision_model.ngeoms:
                o = None
            else:
                o = self.collision_model.geometryObjects[o_id]
            objs.append(o)

        if len(objs) == 1:
            return objs[0]
        else:
            return objs

    def getSignedDistance(self, o1, tf1, o2, tf2):
        o1_geo_type = o1.geometry.getNodeType()
        o2_geo_type = o2.geometry.getNodeType()

        if o1_geo_type == fcl.GEOM_SPHERE and o2_geo_type == fcl.GEOM_SPHERE:
            signed_dist = signed_distance_sphere_sphere(tf1[0], tf2[0], o1.geometry.radius, o2.geometry.radius)
        elif o1_geo_type == fcl.GEOM_HALFSPACE and o2_geo_type == fcl.GEOM_SPHERE:
            # norm and displacement of the dividing plane
            d = o1.geometry.d
            nw = tf1[1] @ o1.geometry.n
            signed_dist = signed_distance_half_space_sphere(d, tf1[0], nw, tf2[0], o2.geometry.radius)
        elif o1_geo_type == fcl.GEOM_SPHERE and o2_geo_type == fcl.GEOM_HALFSPACE:
            # norm and displacement of the dividing plane
            d = o2.geometry.d
            nw = tf2[1] @ o2.geometry.n
            signed_dist = signed_distance_half_space_sphere(d, tf2[0], nw, tf1[0], o1.geometry.radius)
        elif o1_geo_type == fcl.GEOM_CYLINDER and o2_geo_type == fcl.GEOM_SPHERE:
            signed_dist = signed_distance_sphere_cylinder(tf2[0], tf1[0], o2.geometry.radius, o1.geometry.radius)
        elif o1_geo_type == fcl.GEOM_SPHERE and o2_geo_type == fcl.GEOM_CYLINDER:
            signed_dist = signed_distance_sphere_cylinder(tf1[0], tf2[0], o1.geometry.radius, o2.geometry.radius)


        return signed_dist

    def addCollisionObjects(self, geoms):
        """ Add a list of geometry objects to the collision model

        :param geoms: pinocchio GeometryObject
        :return: None
        """
        for geom in geoms:
            self.collision_model.addGeometryObject(geom)

    def addVisualObjects(self, geoms):
        for geom in geoms:
            self.visual_model.addGeometryObject(geom)

    def addCollisionPairs(self, pairs, expand_name=True):
        """ add collision pairs to the model

        :param pairs: list of tuples
        :param expand_name: append _0 to link names if True
        :return: None
        """
        for pair in pairs:
            id1 = self.collision_model.getGeometryId(pair[0] + "_0" if expand_name else pair[0])
            id2 = self.collision_model.getGeometryId(pair[1] + "_0" if expand_name else pair[1])
            self.collision_model.addCollisionPair(pin.CollisionPair(id1, id2))

    def removeAllCollisionPairs(self):
        self.collision_model.removeAllCollisionPairs()

    def addGroundCollisionObject(self):
        # add a ground plane
        ground_placement = pin.SE3.Identity()
        ground_shape = fcl.Halfspace(np.array([0, 0, 1]), 0)
        ground_geom_obj = pin.GeometryObject(
            "ground_0", self.model.frames[0].parent, ground_shape, ground_placement
        )
        ground_geom_obj.meshColor = np.ones((4))

        self.addCollisionObjects([ground_geom_obj])

    def computeDistances(self, q):
        data = self.model.createData()
        geom_data = pin.GeometryData(self.collision_model)
        pin.computeDistances(self.model, data, self.collision_model, geom_data, q)
        return np.array([result.min_distance for result in geom_data.distanceResults])

class CasadiModelInterface:
    def __init__(self, config):
        self.robot = MobileManipulator3D(config)
        self.scene = Scene(config)  
        self.pinocchio_interface = PinocchioInterface(config)
        
        #TODO: set robot init pose
        sdf_type = config.get("sdf_type", None)
        if sdf_type is None:
            sdf_type = "SDF2DNew"
            config_path = parsing.parse_ros_path({"package": "mmseq_run", "path": "config/map/SDF2D.yaml"})
            config_map_default = parsing.load_config(config_path)
            config = parsing.recursive_dict_update(config, config_map_default["controller"])
            print(f"sdf_type was not specified in the config.")

        sdf_class = getattr(map, sdf_type)
        self.sdf_map = sdf_class(config)  
        print(f"Using {sdf_type} Map Model")
        
        if config["sdf_type"][-3:] == "New":
            self.sdf_map_SymMdl = self.sdf_map.sdf_fcn
        else:
            self.sdf_map_SymMdl = CBF('sdf', self.sdf_map, self.sdf_map.dim)

        self.collision_pairs = {"self": [],
                                "static_obstacles": {},
                                "dynamic_obstacles": {}}
        self._setupCollisionPair()

        self.signedDistanceSymMdls = {}             # keyed by collision pair (tuple)
        self.signedDistanceSymMdlsPerGroup = {"static_obstacles": {}, "dynamic_obstacles": {}}
                                                    # nested dictionary, keyed by group name
                                                    # obstacle groups are also a dictionary, keyed by obstacle name
        self._setupSelfCollisionSymMdl()
        self._setupStaticObstaclesCollisionSymMdl()
        self._setupSDFCollisionSymMdl()


    def _addCollisionPairFromTwoGroups(self, group1, group2):
        """ add all possible collision link pairs, one from each group

        :param group1: a list of collision link
        :param group2: another list of collision link
        :return: nested list of possible pairs
        """

        pairs = []
        for link_name1 in group1:
            for link_name2 in group2:
                pairs.append([link_name1, link_name2])

        return pairs

    def _setupCollisionPair(self):
        # base
        self.collision_pairs["self"] = [["ur10_arm_forearm_collision_link", "base_collision_link"]]
        self.collision_pairs["self"] += self._addCollisionPairFromTwoGroups(self.robot.collision_link_names["base"],
                                                                            self.robot.collision_link_names["wrist"] +
                                                                            self.robot.collision_link_names["tool"])
        # upper arm
        self.collision_pairs["self"] += self._addCollisionPairFromTwoGroups(self.robot.collision_link_names["upper_arm"][:2],
                                                                            self.robot.collision_link_names["wrist"] +
                                                                            self.robot.collision_link_names["tool"])
        # forearm
        self.collision_pairs["self"] += self._addCollisionPairFromTwoGroups(self.robot.collision_link_names["forearm"][:2],
                                                                            self.robot.collision_link_names["tool"] +
                                                                            self.robot.collision_link_names["rack"])

        for obstacle in self.scene.collision_link_names.get("static_obstacles", []):
            if obstacle == "ground":
                self.collision_pairs["static_obstacles"][obstacle] = self._addCollisionPairFromTwoGroups([obstacle],
                                                                                                         self.robot.collision_link_names[
                                                                                                             "wrist"] +
                                                                                                         self.robot.collision_link_names[
                                                                                                             "forearm"] +
                                                                                                         self.robot.collision_link_names[
                                                                                                             "upper_arm"])
            else:
                self.collision_pairs["static_obstacles"][obstacle] = self._addCollisionPairFromTwoGroups([obstacle],
                                                                            self.robot.collision_link_names["base"] +
                                                                            self.robot.collision_link_names["wrist"] +
                                                                            self.robot.collision_link_names["forearm"] +
                                                                            self.robot.collision_link_names["upper_arm"])
        
        if self.sdf_map.dim == 2:
            self.collision_pairs["sdf"] = self._addCollisionPairFromTwoGroups(["map"],
                                                                              self.robot.collision_link_names["base"])
        elif self.sdf_map.dim == 3:
            self.collision_pairs["sdf"] = self._addCollisionPairFromTwoGroups(["map"],
                                                                              self.robot.collision_link_names["base"]+
                                                                              self.robot.collision_link_names["wrist"] +
                                                                              self.robot.collision_link_names["forearm"] +
                                                                              self.robot.collision_link_names["upper_arm"] + 
                                                                              self.robot.collision_link_names["tool"])

    def _setupSelfCollisionSymMdl(self):
        sd_syms = []
        for pair in self.collision_pairs["self"]:

            os = self.pinocchio_interface.getGeometryObject(pair)
            if None in os:
                print("either {} or {} isn't a collision geometry".format(*pair))
                continue

            sd_sym = self.pinocchio_interface.getSignedDistance(os[0], self.robot.collisionLinkKinSymMdls[pair[0]](self.robot.q_sym),
                                                                os[1], self.robot.collisionLinkKinSymMdls[pair[1]](self.robot.q_sym))
            sd_syms.append(sd_sym)
            sd_fcn = cs.Function("sd_" + pair[0] + "_" + pair[1], [self.robot.q_sym], [sd_sym])
            self.signedDistanceSymMdls[tuple(pair)] = sd_fcn

        self.signedDistanceSymMdlsPerGroup["self"] = cs.Function("sd_self", [self.robot.q_sym], [cs.vertcat(*sd_syms)])

    def _setupStaticObstaclesCollisionSymMdl(self):
        for obstacle, pairs in self.collision_pairs["static_obstacles"].items():
            sd_syms = []
            for pair in pairs:
                os = self.pinocchio_interface.getGeometryObject(pair)
                if None in os:
                    print("either {} or {} isn't a collision geometry".format(*pair))
                    continue

                sd_sym = self.pinocchio_interface.getSignedDistance(os[0], self.scene.collisionLinkKinSymMdls[pair[0]]([]),
                                                                    os[1], self.robot.collisionLinkKinSymMdls[pair[1]](self.robot.q_sym))
                sd_syms.append(sd_sym)
                sd_fcn = cs.Function("sd_" + pair[0] + "_" + pair[1], [self.robot.q_sym], [sd_sym])
                self.signedDistanceSymMdls[tuple(pair)] = sd_fcn

            self.signedDistanceSymMdlsPerGroup["static_obstacles"][obstacle] = cs.Function("sd_"+obstacle, [self.robot.q_sym], [cs.vertcat(*sd_syms)])

    def _setupSDFCollisionSymMdl(self):
        sd_syms = []
        sdf_map_params_sym = self.sdf_map_SymMdl.mx_in()[1:]        # sdf input [x, param1, param2 ...]
        sdf_map_params_sym_name = self.sdf_map_SymMdl.name_in()[1:]        # sdf input [x, param1, param2 ...]

        for pair in self.collision_pairs["sdf"]:
            o = self.pinocchio_interface.getGeometryObject(pair[1:])
            if o is None:
                print("either {} or {} isn't a collision geometry".format(*pair))
                continue

            pt_sym = self.robot.collisionLinkKinSymMdls[pair[1]](self.robot.q_sym)
            if self.sdf_map.dim == 2:
                sd_sym = self.sdf_map_SymMdl(pt_sym[0][:2], *sdf_map_params_sym) - o.geometry.radius 
            else:
                if pair[1] == self.robot.base_link_name:
                    sd_sym = self.sdf_map_SymMdl(cs.vertcat(pt_sym[0][:2],cs.MX.ones(1)*0.2), *sdf_map_params_sym) - o.geometry.radius 
                else:
                    sd_sym = self.sdf_map_SymMdl(pt_sym[0], *sdf_map_params_sym) - o.geometry.radius 

            sd_syms.append(sd_sym)
            sd_fcn = cs.Function("sd_" + pair[0] + "_" + pair[1], [self.robot.q_sym] + sdf_map_params_sym, [sd_sym],
                                 ["q"] + sdf_map_params_sym_name, ["_".join(["sd(q)"]+pair)])
            self.signedDistanceSymMdls[tuple(pair)] = sd_fcn
        
        self.signedDistanceSymMdlsPerGroup["sdf"] = cs.Function("sd_sdf", [self.robot.q_sym] + sdf_map_params_sym, [cs.vertcat(*sd_syms)],
                                                                          ["q"]+sdf_map_params_sym_name, ["sd(q)"])

    def getSignedDistanceSymMdls(self, name):
        """ get signed distance function by collision link name

        :param name: collision link name
        :return:
        """
        if name == "self":
            return self.signedDistanceSymMdlsPerGroup["self"]
        elif name =="sdf":
            return self.signedDistanceSymMdlsPerGroup["sdf"]
        else:
            for group, name_list in self.scene.collision_link_names.items():
                if name in name_list:
                    return self.signedDistanceSymMdlsPerGroup[group][name]
        print(name + " signed distance function does not exist")
        return None

    def evaluteSignedDistance(self, names:List[str], qs:ndarray, params:Dict[str, List[ndarray]]={}):
        sd = {}
        N = len(qs)
        names.remove("static_obstacles")
        static_obstacle_names = [n for n in self.collision_pairs["static_obstacles"].keys()]
        names += static_obstacle_names
        for name in names:
            sd_fcn = self.getSignedDistanceSymMdls(name)
            sdn_fcn = sd_fcn.map(N, 'thread', 2)
            # sds dimension: num collision pairs x num time step
            if name in static_obstacle_names:
                args = [qs.T] + [p.T for p in params["static_obstacles"]]
            else:
                args = [qs.T] + [p.T for p in params[name]]
            sds = sdn_fcn(*args).toarray()
            sd_mins = np.min(sds, axis=0)
            sd[name] = sd_mins

        return sd
    
    def evaluteSignedDistancePerPair(self, names:List[str], qs:ndarray, params:Dict[str, List[ndarray]]={}):
        sd = {}

        N = len(qs)
        for name in names:
            if name != "static_obstacles":
                sd[name] = {}
                for pair in self.collision_pairs[name]:
                    sd_fcn = self.signedDistanceSymMdls[tuple(pair)]
                    sdn_fcn = sd_fcn.map(N, 'thread', 2)
                    # sds dimension: num collision pairs x num time step
                    args = [qs.T] + [p.T for p in params[name]]
                    sds = sdn_fcn(*args).toarray()
                    # sd_mins = np.min(sds, axis=0)
                    sd[name]["&".join(pair)] = sds.flatten()
            else:
                for obstacle, pairs in self.collision_pairs["static_obstacles"].items():
                    sd[obstacle] = {}
                    for pair in pairs:
                        sd_fcn = self.signedDistanceSymMdls[tuple(pair)]
                        sdn_fcn = sd_fcn.map(N, 'thread', 2)
                        args = [qs.T] + [p.T for p in params["static_obstacles"]]

                        # sds dimension: num collision pairs x num time step
                        sds = sdn_fcn(*args).toarray()
                        # sd_mins = np.min(sds, axis=0)
                        sd[obstacle]["&".join(pair)] = sds.flatten()

        return sd


class Scene:
    def __init__(self, config):
        """ Casadi symbolic model of a 3d Scene

        :param config:
        """
        if config["scene"]["enabled"]:
            urdf_path = parsing.parse_and_compile_urdf(config["scene"]["urdf"])
            urdf = open(urdf_path, 'r').read()
            # we use cas_kin_dyn to build casadi forward kinematics functions
            self.kindyn = cas_kin_dyn.CasadiKinDyn(urdf)  # construct main class
        else:
            self.kindyn = None

        self.collision_link_names = config["scene"]["collision_link_names"]
        self._setupCollisionLinkKinSymMdl()

    def _setupCollisionLinkKinSymMdl(self):
        self.collisionLinkKinSymMdls = {}

        for group, name_list in self.collision_link_names.items():
            for name in name_list:
                if name == "ground":
                    self.collisionLinkKinSymMdls[name] = cs.Function('fk_ground', [cs.SX.sym('empty',0)],
                                                                     [cs.DM.zeros(3), cs.DM.eye(3)])
                else:
                    f = cs.Function.deserialize(self.kindyn.fk(name))
                    self.collisionLinkKinSymMdls[name] = f


class MobileManipulator3D:

    def __init__(self, config):
        """ Casadi symbolic model of Mobile Manipulator

        """
        urdf_path = parsing.parse_and_compile_urdf(config["robot"]["urdf"])
        urdf = open(urdf_path, 'r').read()
        # we use cas_kin_dyn to build casadi forward kinematics functions
        self.kindyn = cas_kin_dyn.CasadiKinDyn(urdf)  # construct main class

        self.numjoint = self.kindyn.nq()
        self.DoF = self.numjoint + 3
        self.dt = config["dt"]
        self.ub_x = parsing.parse_array(config["robot"]["limits"]["state"]["upper"])
        self.lb_x = parsing.parse_array(config["robot"]["limits"]["state"]["lower"])
        self.ub_u = parsing.parse_array(config["robot"]["limits"]["input"]["upper"])
        self.lb_u = parsing.parse_array(config["robot"]["limits"]["input"]["lower"])
        self.ub_udot = parsing.parse_array(config["robot"]["limits"]["input_rate"]["upper"])
        self.lb_udot = parsing.parse_array(config["robot"]["limits"]["input_rate"]["lower"])

        self.link_names = config["robot"]["link_names"]
        self.tool_link_name = config["robot"]["tool_link_name"]
        self.base_link_name = config["robot"]["base_link_name"]
        self.collision_link_names = config["robot"]["collision_link_names"]

        self.qb_sym = cs.MX.sym('qb', 3)
        self.qa_sym = cs.MX.sym('qa', self.numjoint)
        self.q_sym = cs.vertcat(self.qb_sym, self.qa_sym)

        # create self.kinSymMdls dict:{robot links name: cs function of its forward kinematics function}
        self._setupRobotKinSymMdl()
        # create self.collisionLinkKinSymMdls dict:{collision links name: cs function of its forward kinematics function}
        self._setupCollisionLinkKinSymMdl()
        # create self.ssSymMdl robot's state space symbolic model
        self._setupSSSymMdlDI()
        # create self.jacSymMdls dict:{robot links name: cs functions of its jacobian}
        self._setupJacobianSymMdl()
        # create self.manipulability_fcn
        self._setupManipulabilitySymMdl()

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
                         "lb_u": list(self.lb_u),
                         "ub_udot": list(self.ub_udot),
                         "lb_udot": list(self.lb_udot)}

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
        self.ssSymMdl["fmdlk"] = self._discretizefmdl(self.ssSymMdl, self.dt)

    def _setupRobotKinSymMdl(self):
        """ Create kinematic symbolic model for MM links keyed by link name

        """
        self.kinSymMdls = {}
        for name in self.link_names:
            self.kinSymMdls[name] = self._getFk(name)

    def _setupCollisionLinkKinSymMdl(self):
        """ Create kinematic symbolic model for collision links

        :return:
        """
        self.collisionLinkKinSymMdls = {}

        for collision_group, link_list in self.collision_link_names.items():
            for name in link_list:
                self.collisionLinkKinSymMdls[name] = self._getFk(name)

    def _setupJacobianSymMdl(self):
        self.jacSymMdls = {}
        for name in self.link_names:
            fk_fcn = self.kinSymMdls[name]
            fk_pos_eqn, _ = fk_fcn(self.q_sym)
            Jk_eqn = cs.jacobian(fk_pos_eqn, self.q_sym)
            self.jacSymMdls[name] = cs.Function(name + "_jac_fcn", [self.q_sym], [Jk_eqn], ["q"], ["J(q)"])

    def _setupManipulabilitySymMdl(self):
        Jee_fcn = self.jacSymMdls[self.tool_link_name]
        qsym = cs.SX.sym("qsx", self.DoF)
        Jee_eqn = Jee_fcn(qsym)
        man_eqn = cs.det(Jee_eqn @ Jee_eqn.T) ** 0.5

        self.manipulability_fcn = cs.Function("manipulability_fcn", [qsym], [man_eqn])
        arm_man_eqn = cs.det(Jee_eqn[:, 3:] @ Jee_eqn[:, 3:].T) ** 0.5
        self.arm_manipulability_fcn = cs.Function("arm_manipulability_fcn", [qsym], [arm_man_eqn])

    def _getFk(self, link_name, base_frame=False):
        """ Create symbolic function for a link named link_name
            The symbolic function returns the position of its parent joint in and rotation w.r.t the world frame.
            Note this is different from link_state provided by Pybullet which provides CoM position.

        """
        # TODO: Should we handle base through pinocchio by adopting the cartesian base urdf file?
        if link_name == self.base_link_name:
            return cs.Function(link_name + "_fcn", [self.q_sym], [self.qb_sym[:2], self.qb_sym[2]], ["q"], ["pos2", "heading"])

        Hwb = cs.MX.eye(4)
        if not base_frame:
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

    def _discretizefmdl(self, ss_mdl, dt):
        if "linear" in ss_mdl["mdl_type"]:
            x_sym = ss_mdl["x"]
            u_sym = ss_mdl["u"]
            dt_sym = cs.MX.sym("dt")
            A = ss_mdl["A"]
            B = ss_mdl["B"]
            nx = x_sym.size()[0]
            nu = u_sym.size()[0]
            # TODO: discretization time is now hardcoded to 0.1 second. better if we could make dt symbolic too
            M = np.zeros((nx + nu, nx + nu))
            M[:nx, :nx] = A
            M[:nx, nx:] = B
            Md = expm(M * dt)
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
        # For linear system, discreet time model is exact
        if "linear" in ssSymMdl["mdl_type"]:
            fk = ssSymMdl["fmdlk"]
            f_pred = fk.mapaccum(N)
            x_bar = f_pred(xo, u_bar.T)
            x_bar = np.hstack((np.expand_dims(xo, -1), x_bar)).T

        return x_bar

    def checkBounds(self, xs, us, tol=1e-2):
        """

        :param xs:
        :param us:
        :return:
        """

        # check state
        ub_x_check = xs < self.ub_x + tol
        lb_x_check = xs > self.lb_x - tol
        xs_num_violation = np.sum(1 - ub_x_check * lb_x_check, axis=1)
        print(lb_x_check.shape)

        # check input
        ub_u_check = us < self.ub_u + tol
        lb_u_check = us > self.lb_u - tol
        us_num_violation = np.sum(1-ub_u_check * lb_u_check, axis=1)
        print(lb_u_check.shape)

        return xs_num_violation, us_num_violation

    def getEE(self, q, base_frame=False):
        fee = self.kinSymMdls[self.tool_link_name]
        P, rot = fee(q)
        quat = r2q(np.array(rot), order="xyzs")
        if base_frame:
            P[:2] -= q[:2]
            Rwb = rotz(q[2])
            P = Rwb.T @ P


        return P.toarray().flatten(), quat

def verify_link_transforms(robot_sim, sysMdls, link_names):
    for name in link_names:
        print(name)
        q, v = robot_sim.joint_states()

        if name in robot_sim.links:
            link_idx = robot_sim.links[name][0]
        else:
            continue
        pos_sim, orn_sim = robot_sim.link_pose(link_idx)
        # rot_sim = SO3.from_quaternion(orn_sim, 'xyzw').as_matrix()
        rot_sim = q2r(np.array(orn_sim), order="xyzs")
        J_sim = robot_sim.jacobian(q)

        fk_fcn = sysMdls[name]
        # yappi.set_clock_type("wall")
        # yappi.start()
        pos_mdl, rot_mdl = fk_fcn(q)
        # J_fcn = robot.jacSymMdls[name]
        # J_mdl = J_fcn(q)
        # yappi.get_func_stats().print_all()

        pos_mdl = pos_mdl.toarray().flatten()
        # Note that position differences won't be zero because pybullet gives CoM position whereas casadi_kin_dyn gives joint position
        # The differences should be exactly the CoM position in world frame
        print("pos diff:{}, rot diff{}".format(np.linalg.norm(pos_sim - pos_mdl), np.linalg.norm(rot_mdl - rot_sim)))
        # print("J diff{}".format(np.linalg.norm(J_mdl - J_sim[:3])))

def verify_link_jacobian(robot_sim, sysMdls, link_names):
    for name in link_names:
        print(name)
        q, v = robot_sim.joint_states()

        J_sim = robot_sim.jacobian(q)

        J_fcn = sysMdls[name]
        J_mdl = J_fcn(q)

        print("J diff{}".format(np.linalg.norm(J_mdl - J_sim[:3])))


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
    link_names = robot.link_names[1:]
    collision_link_names = []
    for _, collision_link_name in robot.collision_link_names.items():
        collision_link_names += collision_link_name

    # verify robot link transforms and jacobians
    verify_link_transforms(mm, robot.kinSymMdls, link_names)
    verify_link_jacobian(mm, robot.jacSymMdls, link_names)

    # verify collision link transforms (world frame)
    verify_link_transforms(mm, robot.collisionLinkKinSymMdls, collision_link_names)

    # verify collision link transforms (base frame)
    # mm.reset_joint_configuration([0]*9)
    # mm.command_velocity(np.zeros(9))
    # sim.settle(1.0)
    # verify_link_transforms(mm, robot.kinSymMdlsBaseFrame, collision_link_names)

    print("Testing Signed Distance Model")
    q = np.zeros(9)
    q[2] = 1.5
    sd_pin = robot.pinocchio_interface.computeDistances(q[3:])
    sd_sym = robot.collisionSymMdl(q)
    print("Signed Distance by pinocchio: {} by sym mdl: {}".format(sd_pin, sd_sym))
    print("Singed Distance Diff: {}".format(np.linalg.norm(sd_pin - sd_sym)))


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

def test_obstacle_mdl(args):
    # load configuration
    config = parsing.load_config(args.config)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    # Create Sym Mdl
    scene = Scene(ctrl_config)

    for link, f in scene.collisionLinkKinSymMdls.items():
        p, rot = f([])
        print("Link Name: {},R:{}, p:{}".format(link, p, rot))

def test_robot_mdl(args):
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
    link_names = robot.link_names[1:]

    # verify robot link transforms and jacobians
    verify_link_transforms(mm, robot.kinSymMdls, link_names)
    verify_link_jacobian(mm, robot.jacSymMdls, link_names)

    # verify collision link transforms (world frame)
    collision_link_names = []
    for _, collision_link_name in robot.collision_link_names.items():
        collision_link_names += collision_link_name
    verify_link_transforms(mm, robot.collisionLinkKinSymMdls, collision_link_names)

    # verify collision link transforms (base frame)
    # mm.reset_joint_configuration([0]*9)
    # mm.command_velocity(np.zeros(9))
    # sim.settle(1.0)
    # verify_link_transforms(mm, robot.kinSymMdlsBaseFrame, collision_link_names)
    # verify manipulability
    m = robot.manipulability_fcn(mm.home)
    print("Testing motion model integrator")
    dt = 0.1
    a = 1.
    N = 10
    u_bar = np.array([[a] * 9] * N)
    xo = np.array(mm.joint_states()).flatten()
    x_bar_sym = MobileManipulator3D.ssIntegrate(dt, xo, u_bar, robot.ssSymMdl)

    x_bar_num = np.zeros((N + 1, 18))
    x_bar_num[0] = xo
    for k in range(N):
        x_bar_num[k + 1, 9:] = x_bar_num[k, 9:] + a * dt
        x_bar_num[k + 1, :9] = x_bar_num[k, :9] + x_bar_num[k, 9:] * dt + 0.5 * a * dt * dt

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




def test_pinocchio_interface(args):
    # load configuration
    config = parsing.load_config(args.config)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    # Create Sym Mdl
    robot = PinocchioInterface(ctrl_config)
    q = pin.randomConfiguration(robot.model)
    robot.visualize(q)
    input()

def check_maximum_reach(args):
    config = parsing.load_config(args.config)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    # Create Sym Mdl
    robot = MobileManipulator3D(ctrl_config)
    fee = robot.kinSymMdls[robot.tool_link_name]
    q_home = [ 0., -0, 0., 0.5*np.pi, -0.25*np.pi, 0.5*np.pi, -0.25*np.pi, 0.5*np.pi, 0.417*np.pi ]
    q_straight = [0, 0, 0, 0.5*np.pi, 0., 0., 0., 0.5*np.pi, 0.]
    q_upright = [0, 0, 0, 0.5*np.pi, -0.5*np.pi, 0, -0, 0.5*np.pi, 0]

    print("EE position under different configuration")
    print("home: {}".format(fee(q_home)[0]))
    print("straight front: {}".format(fee(q_straight)[0]))
    print("upright: {}".format(fee(q_upright)[0]))


def test_casadi_interface(args):
    # load configuration
    config = parsing.load_config(args.config)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    sym_model = CasadiModelInterface(ctrl_config)

    # q = pin.randomConfiguration(sym_model.pinocchio_interface.model)
    q = np.zeros(6)
    print('----- Self Collision Check -----')
    sym_model.pinocchio_interface.addCollisionPairs(sym_model.collision_pairs["self"])
    sd_fcn = sym_model.getSignedDistanceSymMdls("self")
    self_distance_mdl = sd_fcn(np.hstack((np.zeros(3), q)))
    self_distance_pin = sym_model.pinocchio_interface.computeDistances(q)
    print(self_distance_mdl)
    print(self_distance_mdl - self_distance_pin)

    print('----- Static Obstacle Collision Check -----')
    for obstacle, pairs in sym_model.collision_pairs["static_obstacles"].items():
        sym_model.pinocchio_interface.removeAllCollisionPairs()
        sym_model.pinocchio_interface.addCollisionPairs(pairs)
        sd_fcn = sym_model.getSignedDistanceSymMdls(obstacle)
        self_distance_mdl = sd_fcn(np.hstack((np.zeros(3), q)))
        self_distance_pin = sym_model.pinocchio_interface.computeDistances(q)
        print("Obstacle {}, Distance Diff: ".format(obstacle, self_distance_mdl - self_distance_pin))


def plot_signed_distance_gradient(c_cylinder, r_sphere, r_cylinder, x_range, y_range, grid_size):
    """ Plot the signed distance and its gradient.

    :param c_cylinder: array-like, shape (3,), center of the cylinder (x, y, z)
    :param r_sphere: float, radius of the sphere
    :param r_cylinder: float, radius of the cylinder
    :param x_range: tuple, (min_x, max_x)
    :param y_range: tuple, (min_y, max_y)
    :param grid_size: int, number of points along each axis
    """
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    grad_x = np.zeros_like(X)
    grad_y = np.zeros_like(Y)
    
    # Define CasADi variables
    c_sphere_sym = cs.MX.sym('c_sphere', 3)
    c_cylinder_sym = cs.MX.sym('c_cylinder', 3)
    r_sphere_sym = cs.MX.sym('r_sphere')
    r_cylinder_sym = cs.MX.sym('r_cylinder')

    # Define the signed distance function symbolically
    signed_distance_sym = signed_distance_sphere_cylinder(c_sphere_sym, c_cylinder_sym, r_sphere_sym, r_cylinder_sym)

    # Calculate the gradient symbolically
    gradient = cs.jacobian(signed_distance_sym, c_sphere_sym)
    
    # Create a CasADi function for the signed distance and its gradient
    signed_distance_fn = cs.Function('signed_distance_fn', [c_sphere_sym, c_cylinder_sym, r_sphere_sym, r_cylinder_sym], [signed_distance_sym])
    gradient_fn = cs.Function('gradient_fn', [c_sphere_sym, c_cylinder_sym, r_sphere_sym, r_cylinder_sym], [gradient[:2]])
    
    for i in range(grid_size):
        for j in range(grid_size):
            c_sphere = np.array([X[i, j], Y[i, j], 0])  # Fix z=0 for 2D plot
            Z[i, j] = signed_distance_fn(c_sphere, c_cylinder, r_sphere, r_cylinder).full().item()
            grad = gradient_fn(c_sphere, c_cylinder, r_sphere, r_cylinder).full().flatten()
            grad_x[i, j] = grad[0]
            grad_y[i, j] = grad[1]

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Signed Distance')
    plt.quiver(X, Y, grad_x, grad_y, color='white')
    plt.xlabel('X Coordinate of Sphere Center')
    plt.ylabel('Y Coordinate of Sphere Center')
    plt.title('Signed Distance and Gradient between Sphere and Cylinder')
    plt.show()


def test_signed_distance_sphere_cylinder():
    # Example usage
    c_cylinder = [0, 0, 0]  # Assume z=0 for cylinder center for simplicity
    r_sphere = 0.26
    r_cylinder = 0.325
    x_range = (-5, 5)
    y_range = (-5, 5)
    grid_size = 100

    plot_signed_distance_gradient(c_cylinder, r_sphere, r_cylinder, x_range, y_range, grid_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, help="Path to configuration file.")
    parser.add_argument(
        "--video",
        nargs="?",
        default=None,
        const="",
        help="Record video. Optionally specify prefix for video directory.",
    )
    args = parser.parse_args()
    # test_robot_mdl(args)
    # test_obstacle_mdl(args)
    # test_pinocchio_interface(args)
    args.config = "/home/tracy/Projects/mm_slam/mm_ws/src/mm_sequential_tasks/mmseq_run/config/simple_experiment.yaml"
    # test_casadi_interface(args)
    test_signed_distance_sphere_cylinder()
    # check_maximum_reach(args)

            
        
        
        
        
        
        
        
        
        
        