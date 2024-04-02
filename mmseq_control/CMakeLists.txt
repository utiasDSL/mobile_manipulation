cmake_minimum_required(VERSION 3.0.2)
project(mmseq_control)

find_package(catkin REQUIRED)

catkin_python_setup()
catkin_package()
catkin_install_python(PROGRAMS
            scripts/map_tester.py
            scripts/sdf_tester.py
            DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
          )