cmake_minimum_required(VERSION 3.0)

project(lumina LANGUAGES C CXX CUDA)

find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++17 -fopenmp -O3 -Wall")
set(CMAKE_CUDA_FLAGS "-O3 -Wall")

add_executable(navi ./src/navi/main.cpp ./src/cuda/cuda.cpp ./src/navi/detect.cpp ./src/cuda/detect.cu)
target_include_directories(navi PUBLIC ./include ${PNG_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(navi ${PNG_LIBRARIES} ${JPEG_LIBRARIES} pthread cudart_static)

add_executable(test ./src/test.cpp ./src/cuda/cuda.cpp ./src/cuda/test.cu)
target_include_directories(test PUBLIC ./include ${PNG_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_directories(test PUBLIC /usr/loca/cuda/lib64)
target_link_libraries(test ${PNG_LIBRARIES} ${JPEG_LIBRARIES} pthread cudart_static)
