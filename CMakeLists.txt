cmake_minimum_required(VERSION 3.0.2)
project(mobile_sim)

# Compile as C++11 (ROS Kinetic and newer support this)
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp geometry_msgs sensor_msgs std_msgs
)

############################
# Force Dimension SDK paths
############################
set(FD_SDK_ROOT /home/pieter/sdk-3.7.3)
set(FD_INCLUDE_DIR ${FD_SDK_ROOT}/include)
set(FD_LIB_DIR ${FD_SDK_ROOT}/lib/release/lin-x86_64-gcc)

# Prefer linking directly to the versioned .so to avoid symlink issues.
set(FD_DHD_LIB ${FD_LIB_DIR}/libdhd.so.3.7.3)
set(FD_DRD_LIB ${FD_LIB_DIR}/libdrd.so.3.7.3)

############################
# Include directories
############################
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${FD_INCLUDE_DIR}
)

############################
# Build targets
############################

# C++ node that runs the 4 kHz haptic servo and publishes ROS messages
add_executable(sigma7_haptic_node
  src/sigma7_haptic_node.cpp
)

# Link against catkin + Force Dimension libs
target_link_libraries(sigma7_haptic_node
  ${catkin_LIBRARIES}
  ${FD_DHD_LIB}
  # Uncomment if your node uses DRD (drdc.h / drd* functions):
  # ${FD_DRD_LIB}
)

# Ensure the node can find the Force Dimension .so at runtime without LD_LIBRARY_PATH
# (RPATH is embedded into the built binary)
set_target_properties(sigma7_haptic_node PROPERTIES
  BUILD_RPATH "${FD_LIB_DIR}"
  INSTALL_RPATH "${FD_LIB_DIR}"
)

############################
# Install (optional but recommended)
############################
install(TARGETS sigma7_haptic_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
