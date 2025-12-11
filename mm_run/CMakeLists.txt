cmake_minimum_required(VERSION 3.0.2)
project(mm_run)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED)

catkin_package()

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
   scripts/experiment.py
   nodes/sim_ros.py
   nodes/isaac_sim_ros.py
   nodes/mpc_ros.py
   nodes/planner_test_ros.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
