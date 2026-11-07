cmake_minimum_required(VERSION 3.8)
project(rd_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TagDetection.msg"
  "msg/TagDetectionArray.msg"
  "action/RunPolicy.action"
  "action/GoToLandmark.action"
  "action/Wave.action"
  "action/Say.action"
  "srv/ArmPower.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
