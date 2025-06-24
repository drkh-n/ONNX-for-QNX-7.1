cmake_minimum_required(VERSION 3.10)
project(onnx_inference)

set(CMAKE_CXX_STANDARD 17)

# Use ONNX Runtime path from ENV
if(DEFINED ENV{ONNXRUNTIME_DIR})
    set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_DIR})
else()
    message(FATAL_ERROR "ONNXRUNTIME_DIR environment variable not set")
endif()

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(my_executable src/test_2.cpp src/lodepng.cpp)

target_link_libraries(my_executable onnxruntime)
