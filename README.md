# mm_sequential_tasks
## Installation

**Pinocchio**

First, install the dependencies
```
sudo apt install ros-noetic-eigenpy ros-noetic-hpp-fcl
```
Then, install Pinocchio from source
```
git clone --recursive https://github.com/stack-of-tasks/pinocchio
git checkout master
cd pinocchio && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DPYTHON_EXECUTABLE=/usr/bin/python3 -DBUILD_WITH_COLLISION_SUPPORT=ON
make -j10
make install
```
Configure the environment varaibles. Please check the path you’re adding in PYTHONPATH. Depending on your system, it might use pythonX or pythonX.Y, and site-packages or dist-packages.
```
export PATH=/usr/local/bin:$PATH
export PKG_CONFIG_PATH =/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/lib/pythonX.Y/site-packages:$PYTHONPATH
export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH
```

**Casadi**

I'm using Casadi 3.5.5 PY38. This is an older version. The python pacakge I use for building casadi robot kinematic model `casadi-kin-dyn` doesn't work with the latest casadi version.
To install, download the pacakge from [here](https://web.casadi.org/get/).
Then, update the python path
```
export PYTHONPATH=[Path to your package]:$PYTHONPATH
```

**Acados**

Follow the instructions on their [website](https://docs.acados.org/installation/). Don't forget to install the python interface as well.


**Python Dependencies**

Finally, 
```
python3 -m pip install -r requirements.txt
```

## Run Isaac Sim Simulator Only
1. Make sure [mmseq_sim_isaac](https://github.com/TracyDuX/mmseq_sim_isaac) has been installed.
2. Run
   ```
   roslaunch mmseq_run isaac_sim.launch config:=$(rospack find mmseq_run)/config/3d_collision.yaml isaac-venv:=$ISAACSIM_PYTHON
   ```
   where `$ISAACSIM_PYTHON' is the `./python.sh` file in the Isaac Sim root folder.

## Run controller and Pybullet simulation synchronously in one loop
```
roscd mmseq_run/scripts
python3 experiments.py --config $(rospack find mmseq_run)/config/self_collision_avoidance.yaml --GUI
```
The simulation will put the simulation data and the config file in a folder specified in the config file. To visualize tracking performance,
```
roscd mmseq_utils/scripts
python3 plot_logger_pybullet.py --folder [path_to_data_folder] --tracking
```


## Run controller and Simulation asynchronously in two ROS nodes
To use Pybullet Sim
```
roslaunch mmseq_run run_pybullet_sim.launch config:=$(rospack find mmseq_run)/config/simple_experiment.yaml use_mpc:=True
```
