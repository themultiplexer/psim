CMAKE_MINIMUM_REQUIRED(VERSION 3.18 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenGL REQUIRED)
find_package(OpenCL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

# adjust this based on your available GPUs
# see https://developer.nvidia.com/cuda-gpus
set(CMAKE_CUDA_ARCHITECTURES 61 75)

add_executable(psim main.cpp)
target_link_libraries(psim GL GLEW ${GLUT_LIBRARIES})

