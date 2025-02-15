cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(camera_fusion)

find_package(OpenCV 4.1 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Executable for create matrix exercise
add_executable (2D_feature_tracking src/matching2D_Student.cpp src/MidTermProject_Camera_Student.cpp)
target_link_libraries (2D_feature_tracking ${OpenCV_LIBRARIES})

# Executable for computing stats for task 7
add_executable (task7 src/matching2D_Student.cpp src/task7.cpp)
target_link_libraries (task7 ${OpenCV_LIBRARIES})

# Executable for computing stats for task 8
add_executable (task8 src/matching2D_Student.cpp src/task8.cpp)
target_link_libraries (task8 ${OpenCV_LIBRARIES})

# Executable for computing stats for task 9
add_executable (task9 src/matching2D_Student.cpp src/task9.cpp)
target_link_libraries (task9 ${OpenCV_LIBRARIES})
