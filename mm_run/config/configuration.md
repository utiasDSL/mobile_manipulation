# Experiment Configuration
Configuration files are YAML files typically stored in `mm_run/config/`.

## Including Other YAML Files
Use the `include` key to extend shared configuration files:

```yaml
include:
  - package: "mm_run"
    path: "config/shared.yaml"
```

Included values are overwritten by values in the including file. Multiple includes are supported.

## Top-Level Structure
```yaml
planner    # Task planning (Stack of Tasks)
controller # MPC controller parameters
simulation # Simulation parameters
logging    # Data logging configuration
robot      # Robot model (typically included from config/robot/)
scene      # Environment/scene (typically included from config/scene/)
```

## Planner
The planner section defines tasks executed sequentially by the TaskManager.

### Tasks
Each task in the `tasks` list can specify base and/or end-effector targets. This allows:
- **Simultaneous execution**: Specify both `base_pose` and `ee_pose` in one task
- **Sequential execution**: Specify only `base_pose` or only `ee_pose` in separate tasks

**Common Parameters:**
```yaml
planner:
  tasks:
    - name: str                    # Task identifier
      planner_type: "WaypointPlanner" | "PathPlanner"
      tracking_pos_err_tol: float      # Position error tolerance [m]
      tracking_ori_err_tol: float  # Optional: orientation error tolerance [rad] (default: 0.1)
```

**WaypointPlanner**: Move to target pose(s)
```yaml
# Simultaneous: Move base and EE together
- name: "Approach Target"
  planner_type: "WaypointPlanner"
  base_pose: [x, y, yaw]                    # Optional: SE2 [m, m, rad] in world frame
  ee_pose: [x, y, z, roll, pitch, yaw]      # Optional: SE3 [m, m, m, rad, rad, rad] in world frame
  tracking_pos_err_tol: 0.1
  hold_period: 1.0                          # Optional: hold time at target [s]

# Sequential: Only base moves
- name: "Move Base"
  planner_type: "WaypointPlanner"
  base_pose: [2.0, 1.0, 0.0]
  tracking_pos_err_tol: 0.2

# Sequential: Only EE moves
- name: "Reach Down"
  planner_type: "WaypointPlanner"
  ee_pose: [2.2, 1.2, 0.5, 1.57, 0, 0]
  tracking_pos_err_tol: 0.05
  hold_period: 1.0
```

**PathPlanner**: Follow a pre-computed path
```yaml
# Simultaneous: Follow base and EE paths together
- name: "Coordinated Motion"
  planner_type: "PathPlanner"
  base_path: [[x1, y1, yaw1], [x2, y2, yaw2], ...]  # Optional: Array of SE2 poses
  ee_path: [[x1, y1, z1, r1, p1, y1], [x2, y2, z2, r2, p2, y2], ...]  # Optional: Array of SE3 poses
  dt: 0.1                         # Time step between path points [s]
  tracking_pos_err_tol: 0.1
  end_stop: false                 # Optional: require zero velocity at end (default: false)

# Sequential: Only base path
- name: "Base Path"
  planner_type: "PathPlanner"
  base_path: [[0, 0, 0], [1, 0, 0], [2, 1, 1.57]]
  dt: 0.1
  tracking_pos_err_tol: 0.2

# Sequential: Only EE path
- name: "EE Path"
  planner_type: "PathPlanner"
  ee_path: [[1, 0, 1, 0, 0, 0], [1.5, 0.2, 0.8, 0, 0, 0]]
  dt: 0.1
  tracking_pos_err_tol: 0.02
```

**Note**: At least one of `base_pose`/`base_path` or `ee_pose`/`ee_path` must be specified. Tasks execute sequentially, but each task can coordinate base and EE simultaneously.

## Controller

### Basic Settings

```yaml
controller:
  type: "MPC"                     # Controller type
  dt: 0.1                         # MPC time step [s]
  prediction_horizon: 1.0         # Prediction horizon [s]
  ctrl_rate: 10                   # Controller update rate [Hz]
  cmd_vel_pub_rate: 100           # Command velocity publish rate [Hz]
  cmd_vel_type: "interpolation"   # "integration" or "interpolation"
```

### Collision Avoidance

```yaml
controller:
  # Enable collision avoidance
  self_collision_avoidance_enabled: bool
  static_obstacles_collision_avoidance_enabled: bool
  self_collision_emergency_stop: bool

  # Constraint types
  collision_constraint_type:
    self: "SignedDistanceConstraint"
    static_obstacles: "SignedDistanceConstraint"

  # Safety margins [m]
  collision_safety_margin:
    self: 0.25
    static_obstacles: 0.15

  # Soft constraints
  collision_constraints_softened:
    self: bool
    static_obstacles: bool
```

### Soft Constraints

```yaml
controller:
  soft_cst: bool                  # Enable soft constraints globally
  xu_soft:
    mu: 0.001                     # Penalty weight
    zeta: 0.005                   # Penalty scaling
  collision_soft:
    self: {mu: 0.0001, zeta: 0.005}
    static_obstacles: {mu: 0.0001, zeta: 0.005}
  ee_upward_soft:
    mu: 0.001
    zeta: 0.01
```

### End-Effector Constraints

```yaml
controller:
  ee_upward_constraint_enabled: bool
  ee_upward_deviation_angle_max: 0.20  # [rad]
  ee_pose_tracking_enabled: bool
  base_pose_tracking_enabled: bool
```

### Cost Function Weights

The MPC uses a cost function registry. Available cost functions: `BasePose`, `BaseVel`, `EEPose`, `EEVel`, `ControlEffort`, `Regularization`.

Each cost function uses weights `Qk` (running cost) and `P` (terminal cost):

```yaml
controller:
  cost_params:
    # Base pose (SE2 only: x, y, yaw)
    BasePose:
      Qk: [wx, wy, wyaw]
      P: [px, py, pyaw]

    # Base velocity (2D or 3D)
    BaseVel:
      Qk: [wx, wy] | [wx, wy, wyaw]  # 2D or 3D
      P: [px, py] | [px, py, pyaw]

    # End-effector pose (SE3 only: position + orientation)
    # Set orientation weights to 0 for position-only tracking
    EEPose:
      Qk: [wx, wy, wz, wr, wp, wy]  # [x, y, z, roll, pitch, yaw]
      P: [px, py, pz, pr, pp, py]
    # For base frame tracking, use EEPose with frame="base" (handled internally)

    # End-effector velocity (6D: 3D linear + 3D angular)
    EEVel:
      Qk: [wx, wy, wz, wwx, wwy, wwz]  # [vx, vy, vz, wx, wy, wz]
      P: [px, py, pz, pwx, pwy, pwz]

    # Control effort
    ControlEffort:
      Qqa: [q1, ..., q6]          # Arm joint position weights
      Qqb: [q1, q2, q3]           # Base position weights
      Qva: [v1, ..., v6]          # Arm joint velocity weights
      Qvb: [v1, v2, v3]           # Base velocity weights
      Qua: [u1, ..., u6]          # Arm input weights
      Qub: [u1, u2, u3]           # Base input weights
      Qdua: [du1, ..., du6]       # Arm input rate weights
      Qdub: [du1, du2, du3]       # Base input rate weights

    # Regularization
    Regularization:
      eps: 1.0e-06

    # Slack variables (for soft constraints)
    slack:
      z: 10                       # Linear penalty
      Z: 1                        # Quadratic penalty
```

### Acados Solver Options

```yaml
controller:
  acados:
    name: "MM"                    # Solver identifier
    cython:
      enabled: bool
      recompile: bool
    raise_exception_on_failure: bool
    use_custom_hess: bool
    use_terminal_cost: bool

    ocp_solver_options:
      qp_solver: "FULL_CONDENSING_HPIPM"  # QP solver type
      nlp_solver_type: "SQP_RTI" | "SQP"
      nlp_solver_max_iter: 100
      nlp_solver_tol_comp: 1.e-06
      nlp_solver_tol_stat: 1.0e-03
      nlp_solver_tol_eq: 1.0e-02
      nlp_solver_tol_ineq: 1.0e-02
      qp_solver_iter_max: 100
      qp_solver_warm_start: 2
      integrator_type: "IRK"
      hessian_approx: "GAUSS_NEWTON"
      globalization: "MERIT_BACKTRACKING"
      print_level: 0
      nlp_solver_ext_qp_res: 0

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
  penalize_du: bool      # Penalize input rate in cost
```

### Robot Parameters

```yaml
controller:
  robot:
    dims:
      q: int              # Position dimension
      v: int              # Velocity dimension
      x: int              # State dimension (q + v)
      u: int              # Input dimension
    time_discretization_dt: 0.1
    x0: [q1, ..., qn, v1, ..., vn]

    limits:
      input_rate: {lower: [...], upper: [...]}
      input: {lower: [...], upper: [...]}
      state: {lower: [...], upper: [...]}

    link_names: [str, ...]
    tool_link_name: str
    base_link_name: str
    base_type: "omnidirectional" | "fixed" | "nonholonomic" | "floating"
    tool_vicon_name: str

    collision_link_names:
      base: [str, ...]
      rack: [str, ...]
      upper_arm: [str, ...]
      forearm: [str, ...]
      wrist: [str, ...]
      tool: [str, ...]

    urdf:
      package: str
      path: str
      includes: [str, ...]
      args: {key: value}
```


## Simulation

### Basic Settings

```yaml
simulation:
  timestep: 0.03                  # Simulation timestep [s]
  duration: 25.0                  # Duration [s]
  gravity: [0, 0, -9.81]          # Gravity [m/s²]
  gui: bool                       # Show PyBullet GUI
```

### Robot Configuration

```yaml
simulation:
  robot:
    home: [q1, ..., qn]           # Home joint configuration
    tool_vicon_name: str

    dims:
      q: int
      v: int
      x: int
      u: int

    noise:
      measurement:
        q_std_dev: float
        v_std_dev: float
      process:
        v_std_dev: float

    joint_names: [str, ...]
    link_names: [str, ...]
    tool_joint_name: str
    base_joint_name: str
    tool_link_name: str
    base_link_name: str
    base_type: "omnidirectional" | "fixed" | "nonholonomic" | "floating"

    urdf:
      package: str
      path: str
      includes: [str, ...]
      args: {key: value}
```

### Static Obstacles

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
```

### Dynamic Obstacles

```yaml
simulation:
  dynamic_obstacles:
    enabled: bool
```

### Cameras (Isaac Sim)

```yaml
simulation:
  cameras:
    - name: str
      type: "RGBCamera" | "ToFCamera"
      prim_path: str
      params:
        package: str
        path: str
      translation: [x, y, z]
      orientation: [w, x, y, z]
      ros_topic_name_space: str
```

## Scene

Scene configuration for static obstacles (typically included from `config/scene/`):

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
    enabled: bool
    collision_link_names:
      static_obstacles: [str, ...]
    urdf:
      package: str
      path: str
      includes: [str, ...]
      args:
        obstacle_params_file: str
```


## Logging

```yaml
logging:
  log_dir: str                    # Directory name (relative to results/)
  log_level: int                  # 0=not set, 10=debug, 20=info, 30=warning, 40=error
```

Logs are saved to `mm_run/results/[log_dir]/[TIMESTAMP]/`:
- `combined/` - Synchronous experiments (sim + control in one process)
- `sim/` - Simulation data (asynchronous)
- `control/` - Controller data (asynchronous)
