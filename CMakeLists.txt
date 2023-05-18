cmake_minimum_required(VERSION 3.25)
project(cuda_niewidy CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(cuda_niewidy main.cu)

set_target_properties(cuda_niewidy PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(lib)
add_subdirectory(res)