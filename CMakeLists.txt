cmake_minimum_required(VERSION 3.0)

project(lumina LANGUAGES C CXX)

find_package(PNG REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(autopilot ./src/main.cpp)
target_include_directories(autopilot PUBLIC ./include ${PNG_INCLUDE_DIRS})
target_link_libraries(autopilot ${PNG_LIBRARIES})
