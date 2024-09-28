cmake_minimum_required(VERSION 3.8)
project(ultralytics_ros)
set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/YoloResult.msg"
  DEPENDENCIES
  sensor_msgs
  std_msgs
  vision_msgs
)

install(PROGRAMS
  script/tracker_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  script/utils/postprocessing.py
  DESTINATION lib/${PROJECT_NAME}/utils/
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()