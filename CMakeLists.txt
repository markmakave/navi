cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_COMPILER /usr/bin/g++-11)
set(CMAKE_C_COMPILER /usr/bin/gcc-11)

project(lumina LANGUAGES C CXX)

find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -fopenmp")

add_executable(autopilot ./src/main.cpp)
target_include_directories(autopilot PUBLIC ./include ${PNG_INCLUDE_DIRS})
target_link_libraries(autopilot ${PNG_LIBRARIES} ${JPEG_LIBRARIES} pthread)
