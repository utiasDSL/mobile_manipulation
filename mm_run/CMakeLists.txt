cmake_minimum_required(VERSION 3.0.2)
project(mm_run)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  src/mm_run/scripts/experiment.py
  src/mm_run/nodes/sim_ros.py
  src/mm_run/nodes/isaac_sim_ros.py
  src/mm_run/nodes/mpc_ros.py
  src/mm_run/nodes/planner_test_ros.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
