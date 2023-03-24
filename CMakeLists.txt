cmake_minimum_required(VERSION 3.0)

project(lumina LANGUAGES C CXX CUDA)

find_package(PNG REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++20 -fopenmp -O3 -Wall")

add_library(qoi vendor/qoi/qoi.cpp)
target_include_directories(qoi PUBLIC vendor/qoi)

add_library(lm 
    src/base/image.cpp
    src/base/blas.cpp
    src/navi/detect.cpp
    src/cuda/cuda.cpp
    src/cuda/detect.cu
    src/cuda/descript.cu
    src/cuda/distort.cu
)
target_include_directories(lm PUBLIC
    .
    include
    ${PNG_INCLUDE_DIRS} 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(lm qoi ${PNG_LIBRARIES} ${JPEG_LIBRARIES} pthread cudart_static cuda cublas)

add_executable(navi src/navi/main.cpp)
target_include_directories(navi PUBLIC
    include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(navi qoi ${PNG_LIBRARIES} ${JPEG_LIBRARIES} pthread cuda lm)
target_precompile_headers(navi PUBLIC include/lumina.hpp)
