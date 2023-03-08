cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER /usr/bin/gcc-11)
set(CMAKE_CXX_COMPILER /usr/bin/g++-11)

project(lumina LANGUAGES C CXX CUDA)

find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-fopenmp -O3 -Wall")

add_executable(navi ./src/navi/main.cpp)

target_include_directories(navi PUBLIC ./include ${PNG_INCLUDE_DIRS})
target_link_libraries(navi ${PNG_LIBRARIES} ${JPEG_LIBRARIES} pthread cudart)
target_precompile_headers(navi PUBLIC 
    ./include/base/matrix.hpp
    ./include/base/image.hpp
    ./include/cuda/matrix.cuh    
)
