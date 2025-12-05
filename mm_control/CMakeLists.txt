cmake_minimum_required(VERSION 3.0.2)
project(mm_control)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
            scripts/mpc_generate_c_code.py
            tests/test_constraints.py
            tests/test_cost_functions.py
            tests/test_mpc.py
            tests/test_robot_model.py
            tests/test_SO3.py
            DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
          )
