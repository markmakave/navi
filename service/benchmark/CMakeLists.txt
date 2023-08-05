add_executable(benchmark main.cpp)
target_link_libraries(benchmark lumina)
set_target_properties(benchmark
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)