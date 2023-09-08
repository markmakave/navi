add_executable(reference main.cpp)
target_link_libraries(reference lumina)
set_target_properties(reference
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)