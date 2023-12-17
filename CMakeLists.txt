CMAKE_MINIMUM_REQUIRED(VERSION 3.18 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)
project(psim)

find_package(OpenGL REQUIRED)
find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

# adjust this based on your available GPUs
# see https://developer.nvidia.com/cuda-gpus
set(CMAKE_CUDA_ARCHITECTURES 61 75)

include_directories(${OpenCL_INCLUDE_DIRS})

add_executable(psim main.cpp)

target_link_libraries(psim ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} ${OpenCV_LIBS} OpenCL::OpenCL)

