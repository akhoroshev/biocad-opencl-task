cmake_minimum_required(VERSION 3.1)
project(biocad_opencl_task)

set(CMAKE_CXX_STANDARD 17)

add_executable(biocad_opencl_task main.cpp)
target_link_libraries(biocad_opencl_task OpenCL)
