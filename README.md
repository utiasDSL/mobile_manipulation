# mm_sequential_tasks
## Installation

**Pinocchio**
First, install the dependencies
```bash
sudo apt install libeigen3-dev ros-noetic-eigenpy ros-noetic-hpp-fcl ros-noetic-pinocchio
```

Make sure to also source your ROS environment
```bash
source /opt/ros/noetic/setup.bash
```

Configure the environment varaibles. Please check the path you’re adding in PYTHONPATH. Depending on your system, it might use pythonX or pythonX.Y, and site-packages or dist-packages.
```bash
export PATH=/usr/local/bin:$PATH
export PKG_CONFIG_PATH =/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/lib/pythonX.Y/site-packages:$PYTHONPATH
export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH
```

**Acados**
Follow the instructions on their [website](https://docs.acados.org/installation/). Don't forget to install the python interface as well.

**Python Dependencies**
Finally,
```bash
python3 -m pip install -r requirements.txt
```

## Run Isaac Sim Simulator Only
1. Make sure [mmseq_sim_isaac](https://github.com/TracyDuX/mmseq_sim_isaac) has been installed.
2. Run
   ```bash
   roslaunch mmseq_run isaac_sim.launch config:=$(rospack find mmseq_run)/config/3d_collision.yaml isaac-venv:=$ISAACSIM_PYTHON
   ```
   where `$ISAACSIM_PYTHON' is the `./python.sh` file in the Isaac Sim root folder.

## Compile Controller
```bash
rosrun mmseq_control_new mpc_generate_c_code.py --config $(rospack find mmseq_run)/config/self_collision_avoidance.yaml
```

## Run controller and Pybullet simulation synchronously in one loop
```bash
roscd mmseq_run/scripts
python3 experiments.py --config $(rospack find mmseq_run)/config/self_collision_avoidance.yaml --GUI
```

The simulation will put the simulation data and the config file in a folder specified in the config file. To visualize tracking performance,
```bash
roscd mmseq_utils/scripts
python3 plot_logger_pybullet.py --folder [path_to_data_folder] --tracking
```

## Run controller and Simulation asynchronously in two ROS nodes
To use Pybullet Sim
```bash
roslaunch mmseq_run run_pybullet_sim.launch config:=$(rospack find mmseq_run)/config/simple_experiment.yaml use_mpc:=True
```
