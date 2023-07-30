cmake_minimum_required(VERSION 3.27)

project(lumina LANGUAGES C CXX CUDA)

find_package(PNG REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++20 -O3 -Wall -Wextra -fopenmp")
include_directories(
    .
    include
    ${PNG_INCLUDE_DIRS} 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

add_library(qoi vendor/qoi/qoi.cpp)
target_include_directories(qoi PUBLIC vendor/qoi)

add_library(lumina 
    src/base/image.cpp

    src/slam/descript.cpp

    src/cuda/cuda.cpp
    # src/cuda/blas.cu
    # src/cuda/blas.cpp
    src/cuda/detect.cu
    src/cuda/descript.cu
    src/cuda/distort.cu
    src/cuda/convolve.cu
)
target_link_libraries(lumina qoi ${PNG_LIBRARIES} pthread cudart_static cuda cublas)
set_property(TARGET lumina PROPERTY CUDA_ARCHITECTURES native)

add_executable(navi src/navi/main.cpp)
target_link_libraries(navi qoi ${PNG_LIBRARIES} pthread cuda lumina)
set_property(TARGET navi PROPERTY CUDA_ARCHITECTURES native)
