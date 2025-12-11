# Experiment Configuration

The configuration for each experiment (simulated or hardware) is specified using a YAML file, which are typically stored in `mm_run/config/`.

## Including Other YAML Files

Often only a few configuration parameters change between experiments, so we would like to be able to extend a general shared YAML file with only these few differences. This is done using the `include` key, the value of which is a block sequence; each element of the sequence contains `package`, the name of the ROS package, and `path`, the relative path to the YAML file from that package. The element can also contain the `key` key, which specifies the key that included YAML parameters should be placed under in the overall configuration hierarchy.

For example, suppose I have my YAML file `mine.yaml` and I want to include the parameters from `shared.yaml` located at `mm_run/config/shared.yaml`, where `mm_run` is a ROS package. Then in `mine.yaml` (typically at the top), I write:
```yaml
include:
  - package: "mm_run"
    path: "config/shared.yaml"
```

When one file includes another, any keys present in both take the value from the file doing the including; in other words, the values in the included file are overwritten. You can include as many files into another as desired.

## Parameters

The top-level keys for the mobile manipulation project are:
```yaml
planner    # Task planning parameters (Stack of Tasks)
controller # MPC controller parameters
simulation # Simulation parameters
logging    # How/where to log data
robot      # Robot model parameters (typically included from config/robot/)
scene      # Environment/scene parameters (typically included from config/scene/)
map        # SDF map parameters (typically included from config/map/)
```

## Planner

The planner section defines the Stack of Tasks (SoT) configuration, which manages multiple planning tasks for the mobile manipulator.

### Stack of Tasks Type

```yaml
planner:
  sot_type: "SoTStatic"  # Type of Stack of Tasks manager
                          # Options: "SoTStatic", "SoTCyclic", "SoTDynamic"
```

- **SoTStatic**: Executes tasks sequentially, moving to the next task when the current one finishes
- **SoTCyclic**: Cycles through tasks in a repeating pattern
- **SoTDynamic**: Dynamically reorders tasks based on conditions

### Tasks

Each task in the `tasks` list defines a planning objective. Tasks can be for the mobile base or the end-effector.

#### Common Task Parameters

All planner tasks share these common parameters:

```yaml
planner:
  tasks:
    - name: str                    # Name identifier for this task
      planner_type: str            # Type of planner (see below)
      frame_id: "base" | "EE"      # Reference frame: "base" for mobile base, "EE" for end-effector
      tracking_err_tol: float      # Position tracking error tolerance [m]
```

#### Base Planners

Planners for the mobile base (`frame_id: "base"`):

**BaseSingleWaypoint**: Move base to a single 2D position waypoint
```yaml
- frame_id: "base"
  name: "Base Position"
  planner_type: "BaseSingleWaypoint"
  target_pos: [x, y]              # Target position [m, m]
  tracking_err_tol: 0.2           # Position error tolerance [m]
```

**BaseSingleWaypointPose**: Move base to a single pose (position + orientation)
```yaml
- frame_id: "base"
  name: "Base Pose"
  planner_type: "BaseSingleWaypointPose"
  target_pose: [x, y, yaw]        # Target pose [m, m, rad]
  tracking_err_tol: 0.2           # Pose error tolerance [m, m, rad]
```

**BasePosTrajectoryLine**: Move base along a straight-line trajectory
```yaml
- frame_id: "base"
  name: "Base Trajectory"
  planner_type: "BasePosTrajectoryLine"
  initial_pos: [x, y]             # Starting position [m, m]
  target_pos: [x, y]              # Target position [m, m]
  cruise_speed: 1.0               # Desired speed [m/s]
  tracking_err_tol: 0.2           # Position error tolerance [m]
```

**BasePoseTrajectoryLine**: Move base along a straight-line trajectory with orientation
```yaml
- frame_id: "base"
  name: "Base Pose Trajectory"
  planner_type: "BasePoseTrajectoryLine"
  initial_pose: [x, y, yaw]       # Starting pose [m, m, rad]
  target_pose: [x, y, yaw]        # Target pose [m, m, rad]
  cruise_speed: 1.0               # Desired linear speed [m/s]
  yaw_speed: 0.5                  # Desired yaw speed [rad/s]
  tracking_err_tol: 0.2           # Pose error tolerance [m, m, rad]
```

**BasePosTrajectoryCircle**: Move base along a circular trajectory
```yaml
- frame_id: "base"
  name: "Base Circle"
  planner_type: "BasePosTrajectoryCircle"
  c: [cx, cy]                     # Circle center [m, m]
  r: float                        # Circle radius [m]
  cruise_speed: 1.0              # Desired speed [m/s]
  round: 1                        # Number of rounds
  tracking_err_tol: 0.2           # Position error tolerance [m]
```

**ROSTrajectoryPlanner**: Follow a trajectory received via ROS topic
```yaml
- frame_id: "base"
  name: "ROS Trajectory"
  planner_type: "ROSTrajectoryPlanner"
  cruise_speed: 0.35              # Desired speed [m/s]
  yaw_speed: 0.5                  # Desired yaw speed [rad/s]
  yaw_accel: 0.9                  # Yaw acceleration [rad/s²]
  tracking_err_tol: 0.4           # Position error tolerance [m]
  ref_traj_duration: 1.5         # Reference trajectory duration [s]
```

#### End-Effector Planners

Planners for the end-effector (`frame_id: "EE"`):

**EESimplePlanner**: Move end-effector to a single 3D position waypoint
```yaml
- frame_id: "EE"
  name: "EE Position"
  planner_type: "EESimplePlanner"
  target_pos: [x, y, z]           # Target position [m, m, m]
  hold_period: 3.0                # Time to hold at target [s]
  tracking_err_tol: 0.02          # Position error tolerance [m]
```

**EESimplePlannerBaseFrame**: Move end-effector to a position specified in base frame
```yaml
- frame_id: "EE"
  name: "EE Position BaseFrame"
  planner_type: "EESimplePlannerBaseFrame"
  target_pos: [x, y, z]           # Target position in base frame [m, m, m]
  hold_period: 4.0                # Time to hold at target [s]
  tracking_err_tol: 0.4           # Position error tolerance [m]
```

**EEPoseSE3Waypoint**: Move end-effector to a specific pose (position + orientation)
```yaml
- frame_id: "EE"
  name: "EE Pose"
  planner_type: "EEPoseSE3Waypoint"
  target_pose: [x, y, z, roll, pitch, yaw]  # Target pose [m, m, m, rad, rad, rad]
  hold_period: 1.0                # Time to hold at target [s]
  tracking_err_tol: 0.05          # Pose error tolerance
```

**EEPosTrajectoryCircle**: Move end-effector along a circular trajectory
```yaml
- frame_id: "EE"
  name: "EE Circle"
  planner_type: "EEPosTrajectoryCircle"
  c: [cx, cy, cz]                 # Circle center [m, m, m]
  r: float                        # Circle radius [m]
  T: float                        # Period for one round [s]
  round: 1                        # Number of rounds
  tracking_err_tol: 0.02         # Position error tolerance [m]
```

**EEPosTrajectoryLine**: Move end-effector along a straight-line trajectory
```yaml
- frame_id: "EE"
  name: "EE Line"
  planner_type: "EEPosTrajectoryLine"
  initial_pos: [x, y, z]          # Starting position [m, m, m]
  target_pos: [x, y, z]           # Target position [m, m, m]
  cruise_speed: 0.5               # Desired speed [m/s]
  tracking_err_tol: 0.02         # Position error tolerance [m]
```

## Controller

The controller section configures the Model Predictive Control (MPC) controller.

### Basic Controller Settings

```yaml
controller:
  type: "MPC"                     # Controller type (currently only "MPC" supported)
  dt: 0.1                         # MPC time step [s]
  prediction_horizon: 1.0         # Prediction horizon [s]
  ctrl_rate: 10                   # Controller update rate [Hz]
  cmd_vel_pub_rate: 100           # Command velocity publish rate [Hz]
  cmd_vel_type: "interpolation"   # Command velocity type
                                  # Options: "integration" (integrate from acc_bar)
                                  #          "interpolation" (interpolate vel_bar)
```

### Collision Avoidance

```yaml
controller:
  # Enable/disable collision avoidance types
  self_collision_avoidance_enabled: bool      # Prevent robot from colliding with itself
  sdf_collision_avoidance_enabled: bool        # Use SDF map for collision avoidance
  static_obstacles_collision_avoidance_enabled: bool  # Avoid static obstacles from scene

  # Emergency stop on self-collision
  self_collision_emergency_stop: bool         # Stop immediately on self-collision detection

  # Collision constraint types
  collision_constraint_type:
    self: "SignedDistanceConstraint" | "SignedDistanceConstraintCBF"
    sdf: "SignedDistanceConstraint" | "SignedDistanceConstraintCBF"
    static_obstacles: "SignedDistanceConstraint" | "SignedDistanceConstraintCBF"

  # Safety margins for collision avoidance [m]
  collision_safety_margin:
    self: 0.25                   # Safety margin for self-collision
    sdf: 0.25                    # Safety margin for SDF obstacles
    static_obstacles: 0.15       # Safety margin for static obstacles

  # Enable soft constraints (allow violations with penalty)
  collision_constraints_softend:
    self: bool
    sdf: bool
    static_obstacles: bool

  # CBF (Control Barrier Function) gamma parameter
  collision_cbf_gamma:
    self: 0.9                    # CBF parameter for self-collision
    sdf: 0.3                     # CBF parameter for SDF (typically lower)
    static_obstacles: 0.9         # CBF parameter for static obstacles
```

### Soft Constraints

When constraints are softened, violations are penalized rather than strictly enforced:

```yaml
controller:
  soft_cst: bool                  # Enable soft constraints globally

  # Soft constraint parameters for state/input bounds
  xu_soft:
    mu: 0.001                     # Penalty weight
    zeta: 0.005                   # Penalty scaling

  # Soft constraint parameters for collisions
  collision_soft:
    self:
      mu: 0.0001                  # Penalty weight
      zeta: 0.005                 # Penalty scaling
    sdf:
      mu: 0.0001
      zeta: 0.005
    static_obstacles:
      mu: 0.0001
      zeta: 0.005

  # Soft constraint for end-effector upward constraint
  ee_upward_soft:
    mu: 0.001
    zeta: 0.01
```

### End-Effector Constraints

```yaml
controller:
  ee_upward_constraint_enabled: bool    # Constrain end-effector to point upward
  ee_upward_deviation_angle_max: 0.20  # Maximum deviation angle [rad] (0.26 rad ≈ 15°)
  ee_pose_tracking_enabled: bool       # Enable end-effector pose tracking
  base_pose_tracking_enabled: bool     # Enable base pose tracking
```

### Cost Function Weights

The MPC controller minimizes a weighted combination of costs. Each cost function has weights `Qk` (running cost) and `P` (terminal cost):

```yaml
controller:
  cost_params:
    # End-effector position costs (3D)
    EEPos3:
      Qk: [wx, wy, wz]            # Running cost weights [x, y, z]
      P: [px, py, pz]             # Terminal cost weights [x, y, z]

    # End-effector pose costs (6D: position + orientation)
    EEPose:
      Qk: [wx, wy, wz, wr, wp, wy]  # Running cost weights [x, y, z, roll, pitch, yaw]
      P: [px, py, pz, pr, pp, py]   # Terminal cost weights

    # End-effector position in base frame
    EEPos3BaseFrame:
      Qk: [wx, wy, wz]
      P: [px, py, pz]

    # End-effector pose in base frame
    EEPoseBaseFrame:
      Qk: [wx, wy, wz, wr, wp, wy]
      P: [px, py, pz, pr, pp, py]

    # Base position costs (2D)
    BasePos2:
      Qk: [wx, wy]
      P: [px, py]

    # Base position costs (3D)
    BasePos3:
      Qk: [wx, wy, wz]
      P: [px, py, pz]

    # Base pose costs (SE2: x, y, yaw)
    BasePoseSE2:
      Qk: [wx, wy, wyaw]
      P: [px, py, pyaw]

    # End-effector velocity costs
    EEVel3:
      Qk: [wx, wy, wz]
      P: [px, py, pz]

    # Base velocity costs
    BaseVel2:
      Qk: [wx, wy]
      P: [px, py]

    BaseVel3:
      Qk: [wx, wy, wz]
      P: [px, py, pz]

    # Control effort costs
    Effort:
      Qqa: [q1, ..., q6]          # Arm joint position effort weights
      Qqb: [q1, q2, q3]           # Base position effort weights
      Qva: [v1, ..., v6]          # Arm joint velocity effort weights
      Qvb: [v1, v2, v3]           # Base velocity effort weights
      Qua: [u1, ..., u6]          # Arm joint input effort weights
      Qub: [u1, u2, u3]           # Base input effort weights
      Qdua: [du1, ..., du6]       # Arm joint input rate weights
      Qdub: [du1, du2, du3]       # Base input rate weights

    # Regularization cost
    Regularization:
      eps: 1.0e-06                # Regularization epsilon

    # Slack variable costs (for soft constraints)
    slack:
      z: 10                       # Linear slack penalty
      Z: 1                        # Quadratic slack penalty
```

### Acados Solver Options

The MPC uses Acados for optimization. Configure the solver here:

```yaml
controller:
  acados:
    name: "MM"                    # Solver name identifier
    cython:
      enabled: bool                # Enable Cython code generation
      recompile: bool              # Recompile on each run (set false for repeated runs)
    raise_exception_on_failure: bool  # Raise exception if solver fails
    use_custom_hess: bool         # Use custom Hessian approximation
    use_terminal_cost: bool       # Include terminal cost in optimization

    ocp_solver_options:
      # QP solver options
      qp_solver: "FULL_CONDENSING_HPIPM"  # QP solver type
                                          # Options: "FULL_CONDENSING_HPIPM",
                                          #          "FULL_CONDENSING_QPOASES",
                                          #          "PARTIAL_CONDENSING_HPIPM",
                                          #          "PARTIAL_CONDENSING_QPDUNES",
                                          #          "PARTIAL_CONDENSING_OSQP",
                                          #          "FULL_CONDENSING_DAQP"

      # NLP solver options
      nlp_solver_type: "SQP_RTI" | "SQP"  # NLP solver type
      nlp_solver_max_iter: 100            # Maximum SQP iterations
      nlp_solver_tol_comp: 1.e-06         # Complementarity tolerance
      nlp_solver_tol_stat: 1.0e-03       # Stationarity tolerance
      nlp_solver_tol_eq: 1.0e-02         # Equality constraint tolerance
      nlp_solver_tol_ineq: 1.0e-02       # Inequality constraint tolerance

      # QP solver options
      qp_solver_iter_max: 100            # Maximum QP iterations
      qp_solver_warm_start: 2           # Warm start level (0-2)

      # Integrator options
      integrator_type: "IRK"            # Integrator type (IRK = Implicit Runge-Kutta)
      hessian_approx: "GAUSS_NEWTON"    # Hessian approximation method

      # Globalization
      globalization: "MERIT_BACKTRACKING"  # Globalization strategy

      # Output options
      print_level: 0                    # Solver print level (0 = silent)

      # Residual computation
      nlp_solver_ext_qp_res: 0          # Extended QP residual computation

    # Enable slack variables for different constraint types
    slack_enabled:
      x: bool          # State constraints
      x_e: bool        # Terminal state constraints
      u: bool          # Input constraints
      h_0: bool        # Initial path constraints
      h: bool          # Path constraints
      h_e: bool        # Terminal path constraints
```

### Line Search Parameters

```yaml
controller:
  beta: 0.5              # Line search reduction factor
  alpha: 0.05            # Line search step size
  penalize_du: bool      # Penalize input rate (du) in cost
```

### Robot Parameters (Controller)

```yaml
controller:
  robot:
    dims:
      q: int              # Generalized position dimension
      v: int              # Generalized velocity dimension
      x: int              # State dimension (typically q + v)
      u: int              # Input dimension
    time_discretization_dt: 0.1  # Time discretization for robot model [s]
    x0: [q1, ..., qn, v1, ..., vn]  # Initial state vector
```

### SDF Map Parameters

When using SDF collision avoidance:

```yaml
controller:
  map:
    default_val: 1.8              # Default SDF value (distance) [m]
    map_coverage: [x, y, z]       # Map coverage area [m, m, m]
    voxel_size: 0.2               # Voxel size [m]

    filter_enabled: bool          # Enable map filtering
    filter_type: "tv" | "gaussian"  # Filter type
    guassian_filter_sigma: 10.0   # Gaussian filter sigma (if gaussian)
    tv_filter_weight: 1.0          # Total variation filter weight (if tv)

    offline_map:
      enabled: bool                # Use pre-computed offline map
      path: str | None            # Path to offline map file

  sdf_type: "SDF2D" | "SDF3D"     # SDF dimensionality
```

## Simulation

The simulation section configures the PyBullet or Isaac Sim simulation environment.

### Basic Simulation Settings

```yaml
simulation:
  timestep: 0.03                  # Simulation timestep [s]
  duration: 25.0                   # Simulation duration [s]
  gravity: [0, 0, -9.81]          # Gravity vector [m/s², m/s², m/s²]
  gui: bool                        # Show PyBullet GUI
```

### Robot Configuration (Simulation)

```yaml
simulation:
  robot:
    home: [q1, ..., qn]           # Home joint configuration
    tool_vicon_name: str          # Vicon name for tool tracking

    # System dimensions
    dims:
      q: int                       # Generalized position dimension
      v: int                       # Generalized velocity dimension
      x: int                       # State dimension
      u: int                       # Input dimension

    # Noise models
    noise:
      measurement:
        q_std_dev: float           # Joint position measurement noise std dev
        v_std_dev: float           # Joint velocity measurement noise std dev
      process:
        v_std_dev: float           # Velocity input process noise std dev

    # Joint and link names
    joint_names: [str, ...]        # List of joint names
    link_names: [str, ...]         # List of link names
    tool_joint_name: str           # Tool joint name
    base_joint_name: str           # Base joint name
    tool_link_name: str            # Tool link name
    base_link_name: str             # Base link name
    base_type: "omnidirectional" | "fixed" | "nonholonomic" | "floating"

    # URDF model
    urdf:
      package: str                  # ROS package name
      path: str                    # URDF file path relative to package
      includes: [str, ...]         # List of xacro files to include
      args:
        key: value                 # Xacro argument values
```

### Static Obstacles

```yaml
simulation:
  static_obstacles:
    enabled: bool                  # Add static obstacles to simulation
    urdf:
      package: str                 # ROS package name
      path: str                   # URDF file path
      includes: [str, ...]        # Xacro files to include
      args:
        obstacle_params_file: str  # Path to obstacle parameters file
```

### Dynamic Obstacles

```yaml
simulation:
  dynamic_obstacles:
    enabled: bool                 # Add dynamic obstacles to simulation
```

### Cameras (Isaac Sim)

```yaml
simulation:
  cameras:
    - name: str                    # Camera name
      type: "RGBCamera" | "ToFCamera"  # Camera type
      prim_path: str              # Prim path in Isaac Sim scene
      params:
        package: str               # ROS package for camera params
        path: str                  # Path to camera config file
      translation: [x, y, z]      # Camera translation [m, m, m]
      orientation: [w, x, y, z]   # Camera orientation quaternion
      ros_topic_name_space: str   # ROS topic namespace
```

## Robot

Robot configuration is typically included from `config/robot/thing.yaml` or similar files. It defines the robot model parameters for both controller and simulation.

```yaml
controller:
  robot:
    dims:
      q: int                      # Generalized position dimension
      v: int                      # Generalized velocity dimension
      x: int                      # State dimension
      u: int                      # Input dimension

    x0: [q1, ..., qn, v1, ..., vn]  # Initial state

    limits:
      input_rate:
        lower: [float, ...]       # Lower bounds on input rate
        upper: [float, ...]       # Upper bounds on input rate
      input:
        lower: [float, ...]       # Lower bounds on inputs
        upper: [float, ...]       # Upper bounds on inputs
      state:
        lower: [float, ...]       # Lower bounds on states
        upper: [float, ...]       # Upper bounds on states

    link_names: [str, ...]        # List of link names
    tool_link_name: str           # Tool link name
    base_link_name: str           # Base link name
    base_type: "omnidirectional" | "fixed" | "nonholonomic" | "floating"

    tool_vicon_name: str          # Vicon tracking name for tool

    collision_link_names:
      base: [str, ...]            # Base collision link names
      rack: [str, ...]            # Rack collision link names
      upper_arm: [str, ...]       # Upper arm collision link names
      forearm: [str, ...]         # Forearm collision link names
      wrist: [str, ...]           # Wrist collision link names
      tool: [str, ...]            # Tool collision link names

    urdf:
      package: str                # ROS package name
      path: str                   # URDF file path
      includes: [str, ...]        # Xacro files to include
      args:
        key: value                # Xacro arguments
```

## Scene

Scene configuration defines the environment and obstacles. Typically included from `config/scene/` files.

```yaml
simulation:
  static_obstacles:
    enabled: bool
    urdf:
      package: str
      path: str
      includes: [str, ...]
      args:
        obstacle_params_file: str

controller:
  scene:
    enabled: bool                 # Enable scene for controller
    collision_link_names:
      static_obstacles: [str, ...]  # Collision link names for static obstacles
    urdf:
      package: str
      path: str
      includes: [str, ...]
      args:
        obstacle_params_file: str
```

## Map

Map configuration defines Signed Distance Field (SDF) parameters for collision avoidance. Typically included from `config/map/` files.

```yaml
controller:
  map:
    default_val: float            # Default SDF value [m]
    map_coverage: [x, y, z]      # Map coverage area [m, m, m]
    voxel_size: float            # Voxel size [m]
    filter_enabled: bool         # Enable map filtering
    filter_type: "tv" | "gaussian"
    guassian_filter_sigma: float
    tv_filter_weight: float
    offline_map:
      enabled: bool
      path: str | None

  sdf_type: "SDF2D" | "SDF3D"
```

## Logging

The logging section configures data logging for experiments.

```yaml
logging:
  log_dir: str                    # Directory name for logs (relative to results/)
  log_level: int                  # Python logging level
                                  # 0 = not set, 10 = debug, 20 = info, 30 = warning, 40 = error
```

Logs are saved to `mm_run/results/[log_dir]/[TIMESTAMP]/` with subdirectories:
- `combined/` - For synchronous experiments (sim + control in one process)
- `sim/` - For simulation data (asynchronous)
- `control/` - For controller data (asynchronous)
