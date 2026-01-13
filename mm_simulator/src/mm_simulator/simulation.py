import numpy as np
import pybullet as pyb
import pybullet_data
from pyb_utils.frame import debug_frame_world
from spatialmath.base import rotz

from mm_simulator.camera import VideoManager
from mm_simulator.robot import SimulatedRobot
from mm_utils import math, parsing
from mm_utils.geometry import Box3D


class BulletBody:
    def __init__(
        self,
        mass,
        mu,
        box,
        collision_uid,
        visual_uid,
        position=None,
        orientation=None,
        com_offset=None,
        inertial_orientation=None,
        local_inertia_diagonal=None,
    ):
        """Initialize Bullet body object.

        Args:
            mass (float): Object mass.
            mu (float): Friction coefficient.
            box (Box3D): 3D box geometry.
            collision_uid (int): PyBullet collision shape ID.
            visual_uid (int): PyBullet visual shape ID.
            position (ndarray, optional): Initial position (3,). Defaults to [0, 0, 0].
            orientation (ndarray, optional): Initial orientation quaternion (4,). Defaults to [0, 0, 0, 1].
            com_offset (ndarray, optional): Center of mass offset (3,). Defaults to [0, 0, 0].
            inertial_orientation (ndarray, optional): Inertial frame orientation quaternion (4,). Defaults to [0, 0, 0, 1].
            local_inertia_diagonal (ndarray, optional): Local inertia diagonal (3,). Defaults to None.
        """
        self.mass = mass
        self.mu = mu
        self.box = box
        self.collision_uid = collision_uid
        self.visual_uid = visual_uid

        if position is None:
            position = np.zeros(3)
        self.r0 = position

        if orientation is None:
            orientation = np.array([0, 0, 0, 1])
        self.q0 = orientation

        if com_offset is None:
            com_offset = np.zeros(3)
        self.com_offset = com_offset

        # we need to get the box's orientation correct here for accurate height computation
        C0 = math.quat_to_rot(self.q0)
        self.box.update_pose(rotation=C0)

        if inertial_orientation is None:
            inertial_orientation = np.array([0, 0, 0, 1])
        self.inertial_orientation = inertial_orientation
        self.local_inertia_diagonal = local_inertia_diagonal

    @property
    def height(self):
        """Get object height.

        Returns:
            float: Object height.
        """
        return self.box.height()

    def add_to_sim(self):
        """Actually add the object to the simulation.

        Creates the multi-body object in PyBullet and sets its properties.
        """
        # baseInertialFramePosition is an offset of the inertial frame origin
        # (i.e., center of mass) from the centroid of the object
        # see <https://github.com/erwincoumans/bullet3/blob/d3b4c27db4f86e1853ff7d84185237c437dc8485/examples/pybullet/examples/shiftCenterOfMass.py>
        self.uid = pyb.createMultiBody(
            baseMass=self.mass,
            baseInertialFramePosition=tuple(self.com_offset),
            baseInertialFrameOrientation=tuple(self.inertial_orientation),
            baseCollisionShapeIndex=self.collision_uid,
            baseVisualShapeIndex=self.visual_uid,
            basePosition=tuple(self.r0),
            baseOrientation=tuple(self.q0),
        )

        # update bounding polygon
        C0 = math.quat_to_rot(self.q0)
        self.box.update_pose(self.r0, C0)

        # set friction
        # I do not set a spinning friction coefficient here directly, but let
        # Bullet handle this internally
        pyb.changeDynamics(self.uid, -1, lateralFriction=self.mu)

        # set local inertia if needed (required for objects built from meshes)
        if self.local_inertia_diagonal is not None:
            pyb.changeDynamics(
                self.uid, -1, localInertiaDiagonal=self.local_inertia_diagonal
            )

    def get_pose(self):
        """Get the pose of the object in the simulation.

        Returns:
            tuple: (position, orientation) where position is (3,) array and orientation is (4,) quaternion array.
        """
        pos, orn = pyb.getBasePositionAndOrientation(self.uid)
        return np.array(pos), np.array(orn)

    def get_velocity(self):
        """Get the velocity of the object in the simulation.

        Returns:
            tuple: (linear_velocity, angular_velocity) where each is a (3,) array.
        """
        v, ω = pyb.getBaseVelocity(self.uid)
        return np.array(v), np.array(ω)

    def reset_pose(self, position=None, orientation=None):
        """Reset the pose of the object in the simulation.

        Args:
            position (ndarray, optional): New position (3,). If None, keeps current position.
            orientation (ndarray, optional): New orientation quaternion (4,). If None, keeps current orientation.
        """
        current_pos, current_orn = self.get_pose()
        if position is None:
            position = current_pos
        if orientation is None:
            orientation = current_orn
        pyb.resetBasePositionAndOrientation(self.uid, list(position), list(orientation))

    def change_color(self, rgba):
        """Change the visual color of the object.

        Args:
            rgba (ndarray or tuple): RGBA color (4,) with values in [0, 1].
        """
        pyb.changeVisualShape(self.uid, -1, rgbaColor=list(rgba))

    @staticmethod
    def cylinder(
        mass, mu, radius, height, orientation=None, com_offset=None, color=(0, 0, 1, 1)
    ):
        """Construct a cylinder object.

        Args:
            mass (float): Cylinder mass.
            mu (float): Friction coefficient.
            radius (float): Cylinder radius.
            height (float): Cylinder height.
            orientation (ndarray, optional): Initial orientation quaternion (4,). Defaults to [0, 0, 0, 1].
            com_offset (ndarray, optional): Center of mass offset (3,). Defaults to None.
            color (tuple, optional): RGBA color (4,). Defaults to (0, 0, 1, 1).

        Returns:
            BulletBody: Configured cylinder object.
        """
        if orientation is None:
            orientation = np.array([0, 0, 0, 1])

        # for the cylinder, we rotate by 45 deg about z so that contacts occur
        # aligned with x-y axes
        qz = math.rot_to_quat(rotz(np.pi / 4))
        q = math.quat_multiply(orientation, qz)

        w = np.sqrt(2) * radius
        half_extents = 0.5 * np.array([w, w, height])
        box = Box3D(half_extents)

        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_CYLINDER, radius=radius, height=height
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color
        )
        return BulletBody(
            mass=mass,
            mu=mu,
            box=box,
            collision_uid=collision_uid,
            visual_uid=visual_uid,
            orientation=q,
            com_offset=com_offset,
        )

    @staticmethod
    def cuboid(
        mass, mu, side_lengths, orientation=None, com_offset=None, color=(0, 0, 1, 1)
    ):
        """Construct a cuboid object.

        Args:
            mass (float): Cuboid mass.
            mu (float): Friction coefficient.
            side_lengths (ndarray or list): Side lengths (3,).
            orientation (ndarray, optional): Initial orientation quaternion (4,). Defaults to None.
            com_offset (ndarray, optional): Center of mass offset (3,). Defaults to None.
            color (tuple, optional): RGBA color (4,). Defaults to (0, 0, 1, 1).

        Returns:
            BulletBody: Configured cuboid object.
        """
        half_extents = 0.5 * np.array(side_lengths)
        box = Box3D(half_extents)

        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_BOX, halfExtents=tuple(half_extents)
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_BOX, halfExtents=tuple(half_extents), rgbaColor=color
        )
        return BulletBody(
            mass=mass,
            mu=mu,
            box=box,
            collision_uid=collision_uid,
            visual_uid=visual_uid,
            orientation=orientation,
            com_offset=com_offset,
        )

    @staticmethod
    def sphere(mass, mu, radius, orientation=None, com_offset=None, color=(0, 0, 1, 1)):
        """Construct a sphere object.

        Args:
            mass (float): Sphere mass.
            mu (float): Friction coefficient.
            radius (float): Sphere radius.
            orientation (ndarray, optional): Initial orientation quaternion (4,). Defaults to None.
            com_offset (ndarray, optional): Center of mass offset (3,). Defaults to None.
            color (tuple, optional): RGBA color (4,). Defaults to (0, 0, 1, 1).

        Returns:
            BulletBody: Configured sphere object.
        """
        half_extents = np.ones(3) * radius / 2
        box = Box3D(half_extents)

        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_SPHERE, radius=radius
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_SPHERE, radius=radius, rgbaColor=color
        )
        return BulletBody(
            mass=mass,
            mu=mu,
            box=box,
            collision_uid=collision_uid,
            visual_uid=visual_uid,
            com_offset=com_offset,
        )

    @staticmethod
    def from_config(d, mu, orientation=None):
        """Construct the object from a dictionary.

        Args:
            d (dict): Configuration dictionary with object parameters.
            mu (float): Friction coefficient.
            orientation (ndarray, optional): Initial orientation quaternion (4,). Defaults to None.

        Returns:
            BulletBody: Configured object.

        Raises:
            ValueError: If object shape is not recognized.
        """
        com_offset = np.array(d["com_offset"]) if "com_offset" in d else np.zeros(3)
        if d["shape"] == "cylinder":
            return BulletBody.cylinder(
                mass=d["mass"],
                mu=mu,
                radius=d["radius"],
                height=d["height"],
                color=d["color"],
                orientation=orientation,
                com_offset=com_offset,
            )
        elif d["shape"] == "cuboid":
            return BulletBody.cuboid(
                mass=d["mass"],
                mu=mu,
                side_lengths=d["side_lengths"],
                color=d["color"],
                orientation=orientation,
                com_offset=com_offset,
            )
        else:
            raise ValueError(f"Unrecognized object shape {d['shape']}")


class BulletDynamicObstacle:
    def __init__(
        self, times, positions, velocities, accelerations, radius=0.1, controlled=False
    ):
        """Initialize dynamic obstacle with trajectory modes.

        Args:
            times (list): List of mode transition times.
            positions (list): List of initial positions for each mode (N, 3).
            velocities (list): List of initial velocities for each mode (N, 3).
            accelerations (list): List of accelerations for each mode (N, 3).
            radius (float): Obstacle sphere radius. Defaults to 0.1.
            controlled (bool): If True, use PD control to track trajectory. Defaults to False.
        """
        self.start_time = None
        self._mode_idx = 0
        self.times = times
        self.positions = positions
        self.velocities = velocities
        self.accelerations = accelerations

        self.controlled = controlled
        self.K = 10 * np.eye(3)  # position gain

        self.body = BulletBody.sphere(mass=1, mu=1, radius=radius)
        self.body.r0 = np.array(positions[0])

    @classmethod
    def from_config(cls, config, offset=None):
        """Parse obstacle properties from a config dict.

        Args:
            config (dict): Configuration dictionary with obstacle parameters.
            offset (ndarray, optional): Position offset to apply. Defaults to None.

        Returns:
            BulletDynamicObstacle: Configured dynamic obstacle instance.
        """
        relative = config["relative"]
        if relative and offset is not None:
            offset = np.array(offset)
        else:
            offset = np.zeros(3)

        controlled = config["controlled"]

        times = []
        positions = []
        velocities = []
        accelerations = []
        for mode in config["modes"]:
            times.append(mode["time"])
            positions.append(np.array(mode["position"]) + offset)
            velocities.append(np.array(mode["velocity"]))
            accelerations.append(np.array(mode["acceleration"]))

        return cls(
            times=times,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            radius=config["radius"],
            controlled=controlled,
        )

    def _initial_mode_values(self):
        t = self.times[self._mode_idx]
        if self.start_time is not None:
            t += self.start_time
        r = self.positions[self._mode_idx]
        v = self.velocities[self._mode_idx]
        a = self.accelerations[self._mode_idx]
        return t, r, v, a

    def start(self, t0):
        """Add the obstacle to the simulation.

        Args:
            t0 (float): Simulation time at which to start the obstacle.
        """
        self.start_time = t0
        self.body.add_to_sim()
        v0 = self._initial_mode_values()[2]
        pyb.resetBaseVelocity(self.body.uid, linearVelocity=list(v0))

    def _desired_state(self, t):
        t0, r0, v0, a0 = self._initial_mode_values()
        dt = t - t0
        rd = r0 + dt * v0 + 0.5 * dt**2 * a0
        vd = v0 + dt * a0
        return rd, vd

    def joint_state(self):
        """Get the joint state (position, velocity, acceleration) of the obstacle.

        If the obstacle has not yet been started, the nominal starting state is
        given.

        Returns:
            tuple: (position, velocity, acceleration) where each is a (3,) array.
        """
        if self.start_time is None:
            _, r, v, a = self._initial_mode_values()
        else:
            r = self.body.get_pose()[0]
            v = self.body.get_velocity()[0]
            a = self.accelerations[self._mode_idx]
        return r, v, a

    def step(self, t):
        """Step the object forward in time.

        Args:
            t (float): Current simulation time.

        Returns:
            bool: True if obstacle was reset to a new mode, False otherwise.
        """
        # no-op if obstacle hasn't been started
        reset = False
        if self.start_time is None:
            return reset

        # reset the obstacle if we've stepped into a new mode
        if self._mode_idx < len(self.times) - 1:
            if t - self.start_time >= self.times[self._mode_idx + 1]:
                self._mode_idx += 1
                _, r0, v0, _ = self._initial_mode_values()
                pyb.resetBasePositionAndOrientation(
                    self.body.uid, list(r0), [0, 0, 0, 1]
                )
                pyb.resetBaseVelocity(self.body.uid, linearVelocity=list(v0))
                reset = True

        # velocity needs to be reset at each step of the simulation to negate
        # the effects of gravity
        if self.controlled:
            rd, vd = self._desired_state(t)
            r, _ = self.body.get_pose()
            cmd_vel = self.K @ (rd - r) + vd
            pyb.resetBaseVelocity(self.body.uid, linearVelocity=list(cmd_vel))
        return reset


class BulletSimulation:
    def __init__(self, config, timestamp, cli_args=None):
        """Initialize PyBullet simulation environment.

        Args:
            config (dict): Configuration dictionary with simulation parameters.
            timestamp (datetime): Timestamp for the simulation session.
            cli_args (argparse.Namespace, optional): Command-line arguments. Defaults to None.
        """
        self.config = config

        self.timestep = config["timestep"]
        self.duration = config["duration"]

        if config["gui"]:
            pyb.connect(pyb.GUI, options="--width=1280 --height=720")
        else:
            pyb.connect(pyb.DIRECT)

        pyb.setGravity(*config["gravity"])
        pyb.setTimeStep(self.timestep)

        pyb.resetDebugVisualizerCamera(
            cameraDistance=4,
            cameraYaw=42,
            cameraPitch=-35.8,
            cameraTargetPosition=[1.28, 0.045, 0.647],
        )

        # get rid of extra parts of the GUI
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)

        # setup ground plane
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pyb.loadURDF("plane.urdf", [0, 0, 0])

        # setup robot
        self.robot = SimulatedRobot(config)
        self.robot.reset_joint_configuration(self.robot.home)

        # setup obstacles
        if config["static_obstacles"]["enabled"]:
            urdf_path = parsing.parse_and_compile_urdf(
                config["static_obstacles"]["urdf"]
            )
            obstacles_uid = pyb.loadURDF(parsing.parse_path(urdf_path))
            pyb.changeDynamics(obstacles_uid, -1, mass=0)  # change to static object

        r_ew_w, Q_we = self.robot.link_pose()

        self.dynamic_obstacles = []
        if self.config["dynamic_obstacles"]["enabled"]:
            for c in self.config["dynamic_obstacles"]["obstacles"]:
                obstacle = BulletDynamicObstacle.from_config(c, offset=r_ew_w)
                self.dynamic_obstacles.append(obstacle)

        # mark world frame
        debug_frame_world(0.5, [0] * 3, line_width=3)

        # mark frame at the initial position
        debug_frame_world(0.2, list(r_ew_w), orientation=Q_we, line_width=3)

        # video recording
        if cli_args is not None and "video" in cli_args and cli_args.video is not None:
            video_name = cli_args.video
            self.video_manager = VideoManager.from_config(
                video_name=video_name, config=config, timestamp=timestamp, r_ew_w=r_ew_w
            )
        else:
            self.video_manager = None

        # ghost objects
        self.ghosts = []

        # used to change color when object goes non-statically stable
        self.static_stable = True

    def settle(self, duration):
        """Run simulation while doing nothing.

        Useful to let objects settle to rest before applying control.

        Args:
            duration (float): Duration to run simulation in seconds.
        """
        t = 0
        while t < duration:
            pyb.stepSimulation()
            t += self.timestep

    def launch_dynamic_obstacles(self, t0=0):
        """Start the dynamic obstacles.

        This adds each obstacle to the simulation at its initial state.

        Args:
            t0 (float): Simulation time at which to start obstacles. Defaults to 0.
        """
        for obstacle in self.dynamic_obstacles:
            obstacle.start(t0=t0)

    def dynamic_obstacle_state(self):
        """Get the state vector of all dynamics obstacles.

        Returns:
            ndarray: Concatenated state vector [r0, v0, a0, r1, v1, a1, ...] for all obstacles.
        """
        if len(self.dynamic_obstacles) == 0:
            return np.array([])

        xs = []
        for obs in self.dynamic_obstacles:
            r, v, a = obs.joint_state()
            x = np.concatenate((r, v, a))
            xs.append(x)
        return np.concatenate(xs)

    def step(self, t):
        """Step the simulation forward one timestep.

        Args:
            t (float): Current simulation time.

        Returns:
            tuple: (next_time, obstacle_reset) where next_time is t + timestep and obstacle_reset indicates if any obstacle was reset.
        """
        obstacle_reset = False
        for ghost in self.ghosts:
            ghost.update()
        for obstacle in self.dynamic_obstacles:
            obstacle_reset = obstacle.step(t) or obstacle_reset

        if self.video_manager is not None:
            self.video_manager.record(t)

        pyb.stepSimulation()

        return t + self.timestep, obstacle_reset

    def reset(self, robot_home):
        """Reset robot to home configuration.

        Args:
            robot_home (ndarray): Joint positions for home configuration.
        """
        self.robot.reset_joint_configuration(robot_home)
