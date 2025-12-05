cmake_minimum_required(VERSION 3.0.2)
project(mm_utils)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
   scripts/plot_calibration_results.py
   scripts/plot_logger_ros.py
   scripts/plot_logger_pybullet.py
   scripts/plot_path_comparison.py
   scripts/plot_rosbag.py
   scripts/plot_plan.py
   scripts/demo_rosbag_post_processing.py
   scripts/pub_msg_from_bag.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
