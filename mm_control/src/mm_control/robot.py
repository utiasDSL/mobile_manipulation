from typing import Dict, List

import casadi as cs
import numpy as np
import pinocchio as pin
from numpy import ndarray
from pinocchio.visualize import MeshcatVisualizer as visualizer
from scipy.linalg import expm
from spatialmath.base import r2q, rotz

from mm_utils import parsing

# hppfcl must be imported before casadi_kin_dyn to avoid library conflict
import hppfcl as fcl  # isort: skip
import casadi_kin_dyn.py3casadi_kin_dyn as cas_kin_dyn  # isort: skip


def signed_distance_sphere_sphere(c1, c2, r1, r2):
    """Signed distance between two spheres.

    Args:
        c1 (ndarray or casadi.MX): Center of sphere 1, size 3.
        c2 (ndarray or casadi.MX): Center of sphere 2, size 3.
        r1 (float): Radius of sphere 1.
        r2 (float): Radius of sphere 2.

    Returns:
        casadi.MX or float: Signed distance between the spheres.
    """
    return cs.norm_2(c1 - c2) - r1 - r2


def signed_distance_half_space_sphere(d, p, n, c, r):
    """Signed distance of a sphere to half space.

    Args:
        d (float): Offset from p along n.
        p (ndarray): Offset of the normal vector.
        n (ndarray): Normal vector of the plane.
        c (ndarray or casadi.MX): Center of the sphere, size 3.
        r (float): Radius of the sphere.

    Returns:
        casadi.MX or float: Signed distance of sphere to half space.
    """

    return (c - p).T @ n - d - r


def signed_distance_sphere_cylinder(c_sphere, c_cylinder, r_sphere, r_cylinder):
    """Signed distance between a sphere and a cylinder (with infinite height).

    Args:
        c_sphere (ndarray): Center of the sphere.
        c_cylinder (ndarray): Center of the cylinder.
        r_sphere (float): Radius of the sphere.
        r_cylinder (float): Radius of the cylinder.

    Returns:
        casadi.MX or float: Signed distance between sphere and cylinder.
    """

    return cs.norm_2(c_sphere[:2] - c_cylinder[:2]) - r_sphere - r_cylinder


class PinocchioInterface:
    """Interface to Pinocchio for robot model and collision checking."""

    def __init__(self, config):
        """Initialize Pinocchio interface.

        Args:
            config (dict): Configuration dictionary with robot and scene parameters.
        """
        # 1. build robot model
        urdf_path = parsing.parse_and_compile_urdf(config["robot"]["urdf"])
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path
        )
        # 2. add scene model
        if config["scene"]["enabled"]:
            scene_urdf_path = parsing.parse_and_compile_urdf(config["scene"]["urdf"])
            # here models are passed in so that scene models can be appended to robot model
            pin.buildModelFromUrdf(scene_urdf_path, self.model)
            pin.buildGeomFromUrdf(
                self.model,
                scene_urdf_path,
                pin.GeometryType.COLLISION,
                self.collision_model,
            )
            pin.buildGeomFromUrdf(
                self.model, scene_urdf_path, pin.GeometryType.VISUAL, self.visual_model
            )
        self.addGroundCollisionObject()

        self.collision_link_names = config["robot"]["collision_link_names"].copy()
        if config["scene"]["enabled"]:
            self.collision_link_names.update(config["scene"]["collision_link_names"])

    def visualize(self, q):
        """Visualize robot configuration.

        Args:
            q (ndarray): Joint configuration vector.
        """
        viz = visualizer(self.model, self.collision_model, self.visual_model)
        viz.initViewer(open=True)
        viz.viewer.open()
        viz.loadViewerModel()
        viz.display(q)

    def getGeometryObject(self, link_names):
        """Get geometry objects for given link names.

        Args:
            link_names (list or str): Link name(s) to get geometry objects for.

        Returns:
            GeometryObject or list: Geometry object(s) for the link(s).
        """
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
        """Compute signed distance between two geometry objects.

        Args:
            o1 (GeometryObject): First geometry object.
            tf1 (tuple): Transformation (position, rotation) for object 1.
            o2 (GeometryObject): Second geometry object.
            tf2 (tuple): Transformation (position, rotation) for object 2.

        Returns:
            casadi.MX or float: Signed distance between the objects.
        """
        o1_geo_type = o1.geometry.getNodeType()
        o2_geo_type = o2.geometry.getNodeType()

        if o1_geo_type == fcl.GEOM_SPHERE and o2_geo_type == fcl.GEOM_SPHERE:
            signed_dist = signed_distance_sphere_sphere(
                tf1[0], tf2[0], o1.geometry.radius, o2.geometry.radius
            )
        elif o1_geo_type == fcl.GEOM_HALFSPACE and o2_geo_type == fcl.GEOM_SPHERE:
            # norm and displacement of the dividing plane
            d = o1.geometry.d
            nw = tf1[1] @ o1.geometry.n
            signed_dist = signed_distance_half_space_sphere(
                d, tf1[0], nw, tf2[0], o2.geometry.radius
            )
        elif o1_geo_type == fcl.GEOM_SPHERE and o2_geo_type == fcl.GEOM_HALFSPACE:
            # norm and displacement of the dividing plane
            d = o2.geometry.d
            nw = tf2[1] @ o2.geometry.n
            signed_dist = signed_distance_half_space_sphere(
                d, tf2[0], nw, tf1[0], o1.geometry.radius
            )
        elif o1_geo_type == fcl.GEOM_CYLINDER and o2_geo_type == fcl.GEOM_SPHERE:
            signed_dist = signed_distance_sphere_cylinder(
                tf2[0], tf1[0], o2.geometry.radius, o1.geometry.radius
            )
        elif o1_geo_type == fcl.GEOM_SPHERE and o2_geo_type == fcl.GEOM_CYLINDER:
            signed_dist = signed_distance_sphere_cylinder(
                tf1[0], tf2[0], o1.geometry.radius, o2.geometry.radius
            )

        return signed_dist

    def addCollisionObjects(self, geoms):
        """Add a list of geometry objects to the collision model.

        Args:
            geoms (list): List of pinocchio GeometryObject.
        """
        for geom in geoms:
            self.collision_model.addGeometryObject(geom)

    def addVisualObjects(self, geoms):
        """Add a list of geometry objects to the visual model.

        Args:
            geoms (list): List of pinocchio GeometryObject.
        """
        for geom in geoms:
            self.visual_model.addGeometryObject(geom)

    def addCollisionPairs(self, pairs, expand_name=True):
        """Add collision pairs to the model.

        Args:
            pairs (list): List of tuples of collision pairs.
            expand_name (bool): Append _0 to link names if True.
        """
        for pair in pairs:
            id1 = self.collision_model.getGeometryId(
                pair[0] + "_0" if expand_name else pair[0]
            )
            id2 = self.collision_model.getGeometryId(
                pair[1] + "_0" if expand_name else pair[1]
            )
            self.collision_model.addCollisionPair(pin.CollisionPair(id1, id2))

    def removeAllCollisionPairs(self):
        """Remove all collision pairs from the collision model."""
        self.collision_model.removeAllCollisionPairs()

    def addGroundCollisionObject(self):
        """Add a ground plane collision object."""
        # add a ground plane
        ground_placement = pin.SE3.Identity()
        ground_shape = fcl.Halfspace(np.array([0, 0, 1]), 0)
        ground_geom_obj = pin.GeometryObject(
            "ground_0", self.model.frames[0].parentJoint, ground_shape, ground_placement
        )
        ground_geom_obj.meshColor = np.ones((4))

        self.addCollisionObjects([ground_geom_obj])

    def computeDistances(self, q):
        """Compute distances for all collision pairs.

        Args:
            q (ndarray): Joint configuration vector.

        Returns:
            tuple: (distances, names) where distances is array of minimum distances and names is list of collision pair names.
        """
        data = self.model.createData()
        geom_data = pin.GeometryData(self.collision_model)
        pin.computeDistances(self.model, data, self.collision_model, geom_data, q)
        ds = np.array([result.min_distance for result in geom_data.distanceResults])
        ps = [[cp.first, cp.second] for cp in self.collision_model.collisionPairs]
        names = [
            [
                self.collision_model.geometryObjects[p[0]].name,
                self.collision_model.geometryObjects[p[1]].name,
            ]
            for p in ps
        ]
        return ds, names


class CasadiModelInterface:
    """Interface combining robot, scene, and Pinocchio models for CasADi-based control."""

    def __init__(self, config):
        """Initialize CasADi model interface.

        Args:
            config (dict): Configuration dictionary.
        """
        self.robot = MobileManipulator3D(config)
        self.scene = Scene(config)
        self.pinocchio_interface = PinocchioInterface(config)

        self.collision_pairs = {
            "self": [],
            "static_obstacles": {},
            "dynamic_obstacles": {},
        }
        self.collision_pairs_detailed = {
            "self": [],
            "static_obstacles": {},
            "dynamic_obstacles": {},
        }
        self._setupCollisionPair(config)
        self._setupCollisionPairDetailed()

        self.signedDistanceSymMdls = {}  # keyed by collision pair (tuple)
        self.signedDistanceSymMdlsPerGroup = {
            "static_obstacles": {},
            "dynamic_obstacles": {},
        }
        # nested dictionary, keyed by group name
        # obstacle groups are also a dictionary, keyed by obstacle name
        self._setupSelfCollisionSymMdl()
        self._setupStaticObstaclesCollisionSymMdl()
        self._setupPinocchioCollisionMdl()

    def _addCollisionPairFromTwoGroups(self, group1, group2):
        """Add all possible collision link pairs, one from each group.

        Args:
            group1 (list): List of collision link names.
            group2 (list): Another list of collision link names.

        Returns:
            list: Nested list of possible pairs.
        """

        pairs = []
        for link_name1 in group1:
            for link_name2 in group2:
                pairs.append([link_name1, link_name2])

        return pairs

    def _setupCollisionPair(self, config):
        """Setup collision pairs from configuration.

        Args:
            config (dict): Configuration dictionary.
        """
        if config["robot"].get("collision_pairs", False) and config["robot"][
            "collision_pairs"
        ].get("self", False):
            self.collision_pairs["self"] = config["robot"]["collision_pairs"]["self"]
        else:
            # base
            self.collision_pairs["self"] = [
                ["ur10_arm_forearm_collision_link", "base_collision_link"]
            ]
            self.collision_pairs["self"] += self._addCollisionPairFromTwoGroups(
                self.robot.collision_link_names["base"],
                self.robot.collision_link_names["wrist"]
                + self.robot.collision_link_names["tool"],
            )
            # upper arm
            self.collision_pairs["self"] += self._addCollisionPairFromTwoGroups(
                self.robot.collision_link_names["upper_arm"][:2],
                self.robot.collision_link_names["wrist"]
                + self.robot.collision_link_names["tool"],
            )
            # forearm
            self.collision_pairs["self"] += self._addCollisionPairFromTwoGroups(
                self.robot.collision_link_names["forearm"][:2],
                self.robot.collision_link_names["tool"]
                + self.robot.collision_link_names["rack"],
            )

        for obstacle in self.scene.collision_link_names.get("static_obstacles", []):
            if (
                config["robot"].get("collision_pairs", False)
                and config["robot"]["collision_pairs"].get("static_obstacles", False)
                and config["robot"]["collision_pairs"]["static_obstacles"].get(
                    obstacle, False
                )
            ):
                self.collision_pairs["static_obstacles"][obstacle] = (
                    self._addCollisionPairFromTwoGroups(
                        [obstacle],
                        config["robot"]["collision_pairs"]["static_obstacles"][
                            obstacle
                        ],
                    )
                )
            else:
                if obstacle == "ground":
                    self.collision_pairs["static_obstacles"][obstacle] = (
                        self._addCollisionPairFromTwoGroups(
                            [obstacle], self.robot.collision_link_names["tool"]
                        )
                    )
                else:
                    self.collision_pairs["static_obstacles"][obstacle] = (
                        self._addCollisionPairFromTwoGroups(
                            [obstacle],
                            self.robot.collision_link_names["base"]
                            + self.robot.collision_link_names["wrist"]
                            + self.robot.collision_link_names["forearm"]
                            + self.robot.collision_link_names["upper_arm"],
                        )
                    )

    def _setupCollisionPairDetailed(self):
        """Setup detailed collision pairs for self-collision and obstacles."""
        # base
        self.collision_pairs_detailed["self"] = [
            ["ur10_arm_forearm_collision_link", "base_collision_link"]
        ]
        self.collision_pairs_detailed["self"] += self._addCollisionPairFromTwoGroups(
            self.robot.collision_link_names["base"],
            self.robot.collision_link_names["wrist"]
            + self.robot.collision_link_names["tool"],
        )
        # upper arm
        self.collision_pairs_detailed["self"] += self._addCollisionPairFromTwoGroups(
            self.robot.collision_link_names["upper_arm"],
            self.robot.collision_link_names["wrist"]
            + self.robot.collision_link_names["tool"],
        )
        # forearm
        self.collision_pairs_detailed["self"] += self._addCollisionPairFromTwoGroups(
            self.robot.collision_link_names["forearm"],
            self.robot.collision_link_names["tool"] + ["rack_collision_link"],
        )

        self.collision_pairs_detailed["self"] += self._addCollisionPairFromTwoGroups(
            self.robot.collision_link_names["tool"]
            + self.robot.collision_link_names["wrist"],
            ["rack_collision_link"],
        )

        for obstacle in self.scene.collision_link_names.get("static_obstacles", []):
            if obstacle == "ground":
                self.collision_pairs_detailed["static_obstacles"][obstacle] = (
                    self._addCollisionPairFromTwoGroups(
                        [obstacle],
                        self.robot.collision_link_names["wrist"]
                        + self.robot.collision_link_names["tool"]
                        + self.robot.collision_link_names["forearm"],
                    )
                )
            else:
                self.collision_pairs_detailed["static_obstacles"][obstacle] = (
                    self._addCollisionPairFromTwoGroups(
                        [obstacle],
                        self.robot.collision_link_names["base"]
                        + self.robot.collision_link_names["wrist"]
                        + self.robot.collision_link_names["forearm"]
                        + self.robot.collision_link_names["upper_arm"],
                    )
                )

    def _setupSelfCollisionSymMdl(self):
        """Setup symbolic models for self-collision signed distances."""
        sd_syms = []
        for pair in self.collision_pairs["self"]:
            os = self.pinocchio_interface.getGeometryObject(pair)
            if None in os:
                print(f"either {pair[0]} or {pair[1]} isn't a collision geometry")
                continue

            sd_sym = self.pinocchio_interface.getSignedDistance(
                os[0],
                self.robot.collisionLinkKinSymMdls[pair[0]](self.robot.q_sym),
                os[1],
                self.robot.collisionLinkKinSymMdls[pair[1]](self.robot.q_sym),
            )
            sd_syms.append(sd_sym)
            sd_fcn = cs.Function(
                "sd_" + pair[0] + "_" + pair[1], [self.robot.q_sym], [sd_sym]
            )
            self.signedDistanceSymMdls[tuple(pair)] = sd_fcn

        self.signedDistanceSymMdlsPerGroup["self"] = cs.Function(
            "sd_self", [self.robot.q_sym], [cs.vertcat(*sd_syms)]
        )

    def _setupStaticObstaclesCollisionSymMdl(self):
        """Setup symbolic models for static obstacle signed distances."""
        for obstacle, pairs in self.collision_pairs["static_obstacles"].items():
            sd_syms = []
            for pair in pairs:
                os = self.pinocchio_interface.getGeometryObject(pair)
                if None in os:
                    print(f"either {pair[0]} or {pair[1]} isn't a collision geometry")
                    continue

                sd_sym = self.pinocchio_interface.getSignedDistance(
                    os[0],
                    self.scene.collisionLinkKinSymMdls[pair[0]]([]),
                    os[1],
                    self.robot.collisionLinkKinSymMdls[pair[1]](self.robot.q_sym),
                )
                sd_syms.append(sd_sym)
                sd_fcn = cs.Function(
                    "sd_" + pair[0] + "_" + pair[1], [self.robot.q_sym], [sd_sym]
                )
                self.signedDistanceSymMdls[tuple(pair)] = sd_fcn

            self.signedDistanceSymMdlsPerGroup["static_obstacles"][obstacle] = (
                cs.Function(
                    "sd_" + obstacle, [self.robot.q_sym], [cs.vertcat(*sd_syms)]
                )
            )

    def _setupPinocchioCollisionMdl(self):
        """Setup Pinocchio collision model with collision pairs."""
        self.pinocchio_interface.removeAllCollisionPairs()
        self.pinocchio_interface.addCollisionPairs(
            self.collision_pairs_detailed["self"]
        )
        for obstacle, pairs in self.collision_pairs_detailed[
            "static_obstacles"
        ].items():
            self.pinocchio_interface.addCollisionPairs(pairs)

    def getSignedDistanceSymMdls(self, name):
        """Get signed distance function by collision link name.

        Args:
            name (str): Collision link name.

        Returns:
            casadi.Function or None: Signed distance function, or None if not found.
        """
        if name == "self":
            return self.signedDistanceSymMdlsPerGroup["self"]
        else:
            for group, name_list in self.scene.collision_link_names.items():
                if name in name_list:
                    return self.signedDistanceSymMdlsPerGroup[group][name]
        print(name + " signed distance function does not exist")
        return None

    def evaluateSignedDistance(
        self, names: List[str], qs: ndarray, params: Dict[str, List[ndarray]] = {}
    ):
        """Evaluate signed distances for collision link names.

        Args:
            names (List[str]): List of collision link names to evaluate.
            qs (ndarray): Joint configuration(s), shape (N, nq) or (nq,).
            params (Dict[str, List[ndarray]]): Additional parameters per collision link.

        Returns:
            dict: Dictionary mapping collision link names to signed distance arrays.
        """
        sd = {}
        N = len(qs)
        names.remove("static_obstacles")
        static_obstacle_names = [
            n for n in self.collision_pairs["static_obstacles"].keys()
        ]
        names += static_obstacle_names
        for name in names:
            sd_fcn = self.getSignedDistanceSymMdls(name)
            sdn_fcn = sd_fcn.map(N, "thread", 2)
            # sds dimension: num collision pairs x num time step
            if name in static_obstacle_names:
                args = [qs.T] + [p.T for p in params["static_obstacles"]]
            else:
                args = [qs.T] + [p.T for p in params[name]]
            sds = sdn_fcn(*args).toarray()
            sd_mins = np.min(sds, axis=0)
            sd[name] = sd_mins

        return sd

    def evaluateSignedDistancePerPair(
        self, names: List[str], qs: ndarray, params: Dict[str, List[ndarray]] = {}
    ):
        """Evaluate signed distances per collision pair (not aggregated).

        Args:
            names (List[str]): List of collision link names to evaluate.
            qs (ndarray): Joint configuration(s), shape (N, nq) or (nq,).
            params (Dict[str, List[ndarray]]): Additional parameters per collision link.

        Returns:
            dict: Dictionary mapping collision link names to dictionaries of per-pair signed distances.
        """
        sd = {}

        N = len(qs)
        for name in names:
            if name != "static_obstacles":
                sd[name] = {}
                for pair in self.collision_pairs[name]:
                    sd_fcn = self.signedDistanceSymMdls[tuple(pair)]
                    sdn_fcn = sd_fcn.map(N, "thread", 2)
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
                        sdn_fcn = sd_fcn.map(N, "thread", 2)
                        args = [qs.T] + [p.T for p in params["static_obstacles"]]

                        # sds dimension: num collision pairs x num time step
                        sds = sdn_fcn(*args).toarray()
                        # sd_mins = np.min(sds, axis=0)
                        sd[obstacle]["&".join(pair)] = sds.flatten()

        return sd


class Scene:
    def __init__(self, config):
        """Casadi symbolic model of a 3D Scene.

        Args:
            config (dict): Configuration dictionary.
        """
        if config["scene"]["enabled"]:
            urdf_path = parsing.parse_and_compile_urdf(config["scene"]["urdf"])
            urdf = open(urdf_path, "r").read()
            # we use cas_kin_dyn to build casadi forward kinematics functions
            self.kindyn = cas_kin_dyn.CasadiKinDyn(urdf)  # construct main class
        else:
            self.kindyn = None

        self.collision_link_names = config["scene"].get(
            "collision_link_names", {"static_obstacles": ["ground"]}
        )
        self._setupCollisionLinkKinSymMdl()

    def _setupCollisionLinkKinSymMdl(self):
        """Create kinematic symbolic model for collision links in scene."""
        self.collisionLinkKinSymMdls = {}

        for group, name_list in self.collision_link_names.items():
            for name in name_list:
                if name == "ground":
                    self.collisionLinkKinSymMdls[name] = cs.Function(
                        "fk_ground",
                        [cs.SX.sym("empty", 0)],
                        [cs.DM.zeros(3), cs.DM.eye(3)],
                    )
                else:
                    f = cs.Function.deserialize(self.kindyn.fk(name))
                    self.collisionLinkKinSymMdls[name] = f


class MobileManipulator3D:
    def __init__(self, config):
        """Casadi symbolic model of Mobile Manipulator.

        Args:
            config (dict): Configuration dictionary with robot parameters.
        """
        urdf_path = parsing.parse_and_compile_urdf(config["robot"]["urdf"])
        urdf = open(urdf_path, "r").read()
        # we use cas_kin_dyn to build casadi forward kinematics functions
        self.kindyn = cas_kin_dyn.CasadiKinDyn(urdf)  # construct main class

        self.numjoint = self.kindyn.nq()
        self.DoF = self.numjoint + 3
        self.dt = config["robot"]["time_discretization_dt"]
        self.ub_x = parsing.parse_array(config["robot"]["limits"]["state"]["upper"])
        self.lb_x = parsing.parse_array(config["robot"]["limits"]["state"]["lower"])
        self.ub_u = parsing.parse_array(config["robot"]["limits"]["input"]["upper"])
        self.lb_u = parsing.parse_array(config["robot"]["limits"]["input"]["lower"])
        self.ub_udot = parsing.parse_array(
            config["robot"]["limits"]["input_rate"]["upper"]
        )
        self.lb_udot = parsing.parse_array(
            config["robot"]["limits"]["input_rate"]["lower"]
        )

        self.link_names = config["robot"]["link_names"]
        self.tool_link_name = config["robot"]["tool_link_name"]
        self.base_link_name = config["robot"]["base_link_name"]
        self.collision_link_names = config["robot"]["collision_link_names"]

        self.qb_sym = cs.MX.sym("qb", 3)
        self.qa_sym = cs.MX.sym("qa", self.numjoint)
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
        """Create State-space symbolic model for MM"""
        self.va_sym = cs.MX.sym("va", self.numjoint)
        self.vb_sym = cs.MX.sym(
            "vb", 3
        )  # Assuming nonholonomic vehicle, velocity in world frame
        self.v_sym = cs.vertcat(self.vb_sym, self.va_sym)

        self.x_sym = cs.vertcat(self.q_sym, self.v_sym)
        self.u_sym = cs.MX.sym("u", self.v_sym.size()[0])

        nx = self.x_sym.size()[0]
        nu = self.u_sym.size()[0]
        self.ssSymMdl = {
            "x": self.x_sym,
            "u": self.u_sym,
            "mdl_type": ["linear", "time_invariant"],
            "nx": nx,
            "nu": nu,
            "ub_x": list(self.ub_x),
            "lb_x": list(self.lb_x),
            "ub_u": list(self.ub_u),
            "lb_u": list(self.lb_u),
            "ub_udot": list(self.ub_udot),
            "lb_udot": list(self.lb_udot),
        }

        A = cs.DM.zeros((nx, nx))
        G = cs.DM.eye(self.DoF)
        A[: self.DoF, self.DoF :] = G
        B = cs.DM.zeros((nx, nu))
        B[self.DoF :, :] = cs.DM.eye(nu)

        xdot = A @ self.x_sym + B @ self.u_sym
        fmdl = cs.Function(
            "ss_fcn", [self.x_sym, self.u_sym], [xdot], ["x", "u"], ["xdot"]
        )
        self.ssSymMdl["fmdl"] = fmdl.expand()
        self.ssSymMdl["A"] = A
        self.ssSymMdl["B"] = B
        self.ssSymMdl["fmdlk"] = self._discretizefmdl(self.ssSymMdl, self.dt)

    def _setupRobotKinSymMdl(self):
        """Create kinematic symbolic model for MM links keyed by link name"""
        self.kinSymMdls = {}
        for name in self.link_names:
            self.kinSymMdls[name] = self._getFk(name)

    def _setupCollisionLinkKinSymMdl(self):
        """Create kinematic symbolic model for collision links."""
        self.collisionLinkKinSymMdls = {}

        for collision_group, link_list in self.collision_link_names.items():
            for name in link_list:
                self.collisionLinkKinSymMdls[name] = self._getFk(name)

    def _setupJacobianSymMdl(self):
        """Create jacobian symbolic model for MM links keyed by link name
        Jacobian in the reference frame of the world frame
        For tool link, creates both 3D position Jacobian and 6D spatial Jacobian
        """
        self.jacSymMdls = {}
        for name in self.link_names:
            fk_fcn = self.kinSymMdls[name]
            fk_pos_eqn, fk_rot_eqn = fk_fcn(self.q_sym)
            # Position Jacobian (3D)
            Jk_pos_eqn = cs.jacobian(fk_pos_eqn, self.q_sym)
            self.jacSymMdls[name] = cs.Function(
                name + "_jac_fcn", [self.q_sym], [Jk_pos_eqn], ["q"], ["J(q)"]
            )

            # For tool link, also create 6D spatial Jacobian (linear + angular)
            if name == self.tool_link_name:
                # Compute angular velocity Jacobian from rotation matrix
                # Angular velocity: ω = 0.5 * [R^T * dR/dq_i]_vee for each q_i
                # The vee map extracts the vector from a skew-symmetric matrix
                # For a rotation matrix R, dR/dq gives us the derivative
                # We compute: ω = [R^T * dR/dq]_vee
                # Simplified: compute the skew-symmetric part of R^T * dR/dq
                dR_dq = cs.jacobian(fk_rot_eqn, self.q_sym)
                # dR_dq is 9 x DoF (each column is flattened 3x3 matrix)
                # Reshape each column to 3x3, compute R^T * dR/dq_i, then extract skew part
                Jw_list = []
                for i in range(self.DoF):
                    dR_i = cs.reshape(dR_dq[:, i], 3, 3)
                    # Compute R^T * dR/dq_i
                    R_T_dR = cs.mtimes(fk_rot_eqn.T, dR_i)
                    # Extract skew-symmetric part: ω = [R^T * dR]_vee
                    # For skew-symmetric matrix S, [S]_vee = [S[2,1], S[0,2], S[1,0]]
                    omega_i = (
                        cs.vertcat(
                            R_T_dR[2, 1] - R_T_dR[1, 2],
                            R_T_dR[0, 2] - R_T_dR[2, 0],
                            R_T_dR[1, 0] - R_T_dR[0, 1],
                        )
                        * 0.5
                    )
                    Jw_list.append(omega_i)
                Jw_eqn = cs.horzcat(*Jw_list)
                # Stack linear and angular Jacobians
                J_spatial = cs.vertcat(Jk_pos_eqn, Jw_eqn)
                self.jacSymMdls[name + "_spatial"] = cs.Function(
                    name + "_jac_spatial_fcn",
                    [self.q_sym],
                    [J_spatial],
                    ["q"],
                    ["J_spatial(q)"],
                )

    def _setupManipulabilitySymMdl(self):
        """Setup symbolic models for end-effector and arm manipulability."""
        Jee_fcn = self.jacSymMdls[self.tool_link_name]
        qsym = cs.SX.sym("qsx", self.DoF)
        Jee_eqn = Jee_fcn(qsym)
        man_eqn = cs.det(Jee_eqn @ Jee_eqn.T) ** 0.5

        self.manipulability_fcn = cs.Function("manipulability_fcn", [qsym], [man_eqn])
        arm_man_eqn = cs.det(Jee_eqn[:, 3:] @ Jee_eqn[:, 3:].T) ** 0.5
        self.arm_manipulability_fcn = cs.Function(
            "arm_manipulability_fcn", [qsym], [arm_man_eqn]
        )

    def _getFk(self, link_name, base_frame=False):
        """Create symbolic function for a link named link_name.

        The symbolic function returns the position of its parent joint in and rotation w.r.t the world frame.
        Note this is different from link_state provided by Pybullet which provides CoM position.

        Args:
            link_name (str): Name of the link.
            base_frame (bool): If True, express pose in base frame; if False, in world frame.

        Returns:
            casadi.Function: Forward kinematics function returning (position, rotation).
        """
        # TODO: Should we handle base through pinocchio by adopting the cartesian base urdf file?
        if link_name == self.base_link_name:
            return cs.Function(
                link_name + "_fcn",
                [self.q_sym],
                [self.qb_sym[:2], self.qb_sym[2]],
                ["q"],
                ["pos2", "heading"],
            )

        Hwb = cs.MX.eye(4)  # T related to movement of base
        if not base_frame:
            Hwb[0, 0] = np.cos(self.qb_sym[2])
            Hwb[1, 0] = np.sin(self.qb_sym[2])
            Hwb[0, 1] = -np.sin(self.qb_sym[2])
            Hwb[1, 1] = np.cos(self.qb_sym[2])
            Hwb[:2, 3] = self.qb_sym[:2]

        fk_str = self.kindyn.fk(link_name)
        fk = cs.Function.deserialize(fk_str)
        link_pos, link_rot = fk(self.qa_sym)
        Hbl = cs.MX.eye(4)  # T from base to link
        Hbl[:3, :3] = link_rot
        Hbl[:3, 3] = link_pos
        Hwl = Hwb @ Hbl  # overall transformation

        return cs.Function(
            link_name + "_fcn",
            [self.q_sym],
            [Hwl[:3, 3], Hwl[:3, :3]],
            ["q"],
            ["pos", "rot"],
        )

    def _discretizefmdl(self, ss_mdl, dt):
        """Discretize state-space model for given time step.

        Args:
            ss_mdl (dict): State-space model dictionary.
            dt (float): Discretization time step.

        Returns:
            dict: Discretized state-space model dictionary.
        """
        if "linear" in ss_mdl["mdl_type"]:
            x_sym = ss_mdl["x"]
            u_sym = ss_mdl["u"]
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
            fdsc_fcn = cs.Function(
                "fmdlk", [x_sym, u_sym], [xk1_eqn], ["xk", "uk"], ["xk1"]
            )

        return fdsc_fcn

    @staticmethod
    def ssIntegrate(dt, xo, u_bar, ssSymMdl):
        """Integrate state-space model.

        Args:
            dt (float): Discretization time step.
            xo (ndarray): Initial state.
            u_bar (ndarray): Control inputs, shape [N, nu].
            ssSymMdl (dict): State-space symbolic model.

        Returns:
            ndarray: State trajectory x_bar, shape [N+1, nx].
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
        """Check bounds for states and controls.

        Args:
            xs (ndarray): State trajectory.
            us (ndarray): Control trajectory.
            tol (float): Tolerance for bound checking.

        Returns:
            bool: True if within bounds, False otherwise.
        """

        # check state
        ub_x_check = xs < self.ub_x + tol
        lb_x_check = xs > self.lb_x - tol
        xs_num_violation = np.sum(1 - ub_x_check * lb_x_check, axis=1)
        print(lb_x_check.shape)

        # check input
        ub_u_check = us < self.ub_u + tol
        lb_u_check = us > self.lb_u - tol
        us_num_violation = np.sum(1 - ub_u_check * lb_u_check, axis=1)
        print(lb_u_check.shape)

        return xs_num_violation, us_num_violation

    def getEE(self, q, base_frame=False):
        """Get end-effector position and orientation.

        Args:
            q (ndarray): Joint configuration vector.
            base_frame (bool): If True, express position in base frame; if False, in world frame.

        Returns:
            tuple: (position, quaternion) where position is 3D array and quaternion is 4D array.
        """
        fee = self.kinSymMdls[self.tool_link_name]
        P, rot = fee(q)
        quat = r2q(np.array(rot), order="xyzs")
        if base_frame:
            P[:2] -= q[:2]
            Rwb = rotz(q[2])
            P = Rwb.T @ P

        return P.toarray().flatten(), quat
