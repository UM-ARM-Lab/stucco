cmake_minimum_required(VERSION 3.10)
project(stucco)

find_package(catkin REQUIRED COMPONENTS
    rospy
    std_msgs
)

# no need to install any python code through ROS for now
# catkin_python_setup()

catkin_package()

# catkin_install_python(PROGRAMS
#    scripts/info_theoretic_exploration.py
#    scripts/retrieval_real_main.py
#    scripts/retrieval_main.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )