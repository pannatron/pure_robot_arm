cmake_minimum_required(VERSION 3.8)
project(inverse_perspective_mapping)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(image_transport REQUIRED) # เพิ่มบรรทัดนี้

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(ipm_node src/ipm_node.cpp)
ament_target_dependencies(ipm_node rclcpp sensor_msgs cv_bridge OpenCV image_transport) 
find_package(CURL REQUIRED)  # เพิ่มการหาไลบรารี curl

target_link_libraries(ipm_node ${OpenCV_LIBRARIES} ${CURL_LIBRARIES})  # เพิ่มการลิงก์ curl

install(TARGETS
  ipm_node
  DESTINATION lib/${PROJECT_NAME})

# Add entry points for the Python scripts
install(
  DIRECTORY scripts/
  DESTINATION lib/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)
install(PROGRAMS
  scripts/aruco_detector_node.py
  scripts/aruco_detector2_node.py
  scripts/detect_pun_30_09_2024.py
   scripts/topview.py
   scripts/ros2_command_cil.py
  DESTINATION lib/${PROJECT_NAME}
)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

