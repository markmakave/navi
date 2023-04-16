cmake_minimum_required(VERSION 3.0)

project(lumina LANGUAGES C CXX CUDA)

find_package(PNG REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++17 -fopenmp -O3 -Wall -Wextra")
include_directories(
    .
    include
    ${PNG_INCLUDE_DIRS} 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

add_library(qoi vendor/qoi/qoi.cpp)
target_include_directories(qoi PUBLIC vendor/qoi)

add_library(lm 
    src/base/image.cpp
    src/cuda/blas.cu
    src/cuda/blas.cpp
    src/cuda/cuda.cpp
    src/cuda/detect.cu
    src/cuda/descript.cu
    src/cuda/distort.cu
)
target_link_libraries(lm qoi ${PNG_LIBRARIES} pthread cudart_static cuda cublas)
set_property(TARGET lm PROPERTY CUDA_ARCHITECTURES native)

add_executable(navi src/navi/main.cpp)
target_link_libraries(navi qoi ${PNG_LIBRARIES} pthread cuda lm)
# target_precompile_headers(navi PUBLIC include/lumina.hpp)

add_executable(neural_train test/neural/train.cpp)
target_link_libraries(neural_train lm)

add_executable(neural_infer test/neural/infer.cpp)
target_link_libraries(neural_infer lm)