cmake_minimum_required(VERSION 3.0)

project(lumina LANGUAGES C CXX CUDA)

find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

set(CMAKE_CXX_FLAGS "-fopenmp -O3 -Wall")

add_executable(navi ./src/navi/main.cpp ./src/navi/detect.cpp ./src/cuda/detect.cu)

target_include_directories(navi PUBLIC ./include ${PNG_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(navi ${PNG_LIBRARIES} ${JPEG_LIBRARIES} pthread cudart)

