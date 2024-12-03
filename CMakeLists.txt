cmake_minimum_required(VERSION 3.27)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS -Wall)
set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=undefined -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -fopenmp")

project(lumina LANGUAGES C CXX CUDA)

find_package(PNG REQUIRED)

include_directories(
    .
    include
    ${PNG_INCLUDE_DIRS} 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

add_library(qoi 3rd-party/qoi/qoi.cpp)
target_include_directories(qoi PUBLIC 3rd-party/qoi)

add_library(lumina 
    src/base/image.cpp

    src/slam/fast9.cpp
    src/slam/fast11.cpp
    src/slam/detect.cpp
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

add_executable(navi src/navi/main.cu)
target_link_libraries(navi qoi ${PNG_LIBRARIES} pthread cuda lumina)
set_property(TARGET navi PROPERTY CUDA_ARCHITECTURES native)

# add_subdirectory(service)
