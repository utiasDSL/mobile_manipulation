# Mobile Manipulation
A ROS-based framework for mobile manipulation research, featuring MPC-based control, robot simulation, and planning utilities.

## Package Overview
- **mm_assets**: Robot and scene URDF/mesh files
- **mm_control**: MPC controller implementation using Acados
- **mm_plan**: Planning base classes and simple planners
- **mm_run**: Launch files, configurations, and ROS nodes
- **mm_simulator**: PyBullet simulation interface
- **mm_utils**: Utility functions for math, parsing, logging, etc.

## Installation
### Prerequisites
Ensure you have ROS Noetic installed on your system. Follow the [ROS Noetic installation guide](http://wiki.ros.org/noetic/Installation/Ubuntu) if it's not already set up.

### Installation of `mobile_manipulation_central`
To install `mobile_manipulation_central`, clone this repository into your ROS workspace and compile it using `catkin`.

```bash
cd ~/catkin_ws/src
git clone https://github.com/utiasDSL/mobile_manipulation_central
git checkout mm_dev
cd ~/catkin_ws
catkin build mobile_manipulation_central
source devel/setup.bash
```

### Pinocchio
```bash
sudo apt install libeigen3-dev ros-noetic-eigenpy ros-noetic-hpp-fcl ros-noetic-pinocchio
```

Make sure to source your ROS environment:
```bash
source /opt/ros/noetic/setup.bash
```

### Acados
Follow the instructions on the [Acados website](https://docs.acados.org/installation/). Don't forget to install the Python interface.

### Installing this repo
```bash
cd ~/catkin_ws/src
git clone https://github.com/utiasDSL/mobile_manipulation
cd ~/catkin_ws
catkin build mobile_manipulation
source devel/setup.bash
python3 -m pip install -r requirements.txt
```

## Usage
### Compile MPC Controller
```bash
rosrun mm_control mpc_generate_c_code.py --config $(rospack find mm_run)/config/self_collision_avoidance.yaml
```

### Run Controller with PyBullet Simulation (Synchronous)
```bash
roscd mm_run/scripts
python3 experiments.py --config $(rospack find mm_run)/config/self_collision_avoidance.yaml --GUI
```

### Run Controller and Simulation Asynchronously (ROS Nodes)
```bash
roslaunch mm_run run_pybullet_sim.launch config:=$(rospack find mm_run)/config/simple_experiment.yaml use_mpc:=True
```

### Visualize Results
Results are saved to `mm_run/results/[EXPERIMENT_NAME]/[TIMESTAMP]/` with `sim/` and `control/` subfolders.

```bash
roscd mm_utils/scripts
python3 plot_logger_pybullet.py --folder ../../mm_run/results/[EXPERIMENT_NAME]/[TIMESTAMP]/ --tracking
```

### Isaac Sim (Optional)
If using Isaac Sim, ensure [mm_sim_isaac](https://github.com/TracyDuX/mm_sim_isaac) is installed:
```bash
roslaunch mm_run isaac_sim.launch config:=$(rospack find mm_run)/config/3d_collision.yaml isaac-venv:=$ISAACSIM_PYTHON
```

## Configuration
Configuration files are located in `mm_run/config/`. Key configuration options include:

- **Robot**: Robot model parameters (`config/robot/`)
- **Scene**: Environment and obstacle definitions (`config/scene/`)
- **Controller**: MPC parameters (`config/controller/`)
- **Simulation**: Simulation settings (`config/sim/`)

## Development
This repository provides a foundation for mobile manipulation research. Key extension points:

- **Custom planners**: Extend `Planner` or `TrajectoryPlanner` in `mm_plan/PlanBaseClass.py`
- **Custom cost functions**: Add to `mm_control/MPCCostFunctions.py`
- **Custom constraints**: Add to `mm_control/MPCConstraints.py`
