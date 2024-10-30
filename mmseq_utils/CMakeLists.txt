cmake_minimum_required(VERSION 3.0.2)
project(mmseq_utils)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES mmseq_control
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

catkin_install_python(PROGRAMS
   scripts/plot_calibration_results.py
   scripts/plot_logger_ros.py
   scripts/plot_logger_pybullet.py
   scripts/plot_benchmark_result_squarewave.py
   scripts/plot_benchmark_result_self_collision.py
   scripts/plot_benchmark_result_semi_static_reactive.py
   scripts/plot_path_comparison.py
   scripts/plot_rosbag.py
   scripts/plot_plan.py
   scripts/demo_rosbag_post_processing.py
   scripts/pub_msg_from_bag.py
   scripts/parse_logger_hybrid_controller.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )