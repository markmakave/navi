cmake_minimum_required(VERSION 3.0)

project(lumina LANGUAGES C CXX CUDA)

find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++17 -fopenmp -O3 -Wall")
set(CMAKE_CUDA_FLAGS "-O3 -Wall")

add_library(qoi vendor/qoi/qoi.cpp)
target_include_directories(qoi PUBLIC vendor/qoi)

add_executable(navi 
    src/navi/main.cpp

    src/base/image.cpp
    src/base/blas.cpp

    src/navi/detect.cpp
    
    src/cuda/cuda.cpp
    src/cuda/detect.cu
    src/cuda/descript.cu
)
target_include_directories(navi PUBLIC
    .
    include
    ${PNG_INCLUDE_DIRS} 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(navi qoi ${PNG_LIBRARIES} ${JPEG_LIBRARIES} pthread cudart_static cuda cublas)
