cmake_minimum_required(VERSION 3.0.2)
project(mm_control)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
            scripts/test_constraints.py
            scripts/test_cost_functions.py
            scripts/mpc_generate_c_code.py
            scripts/test_mpc.py
            DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
          )
