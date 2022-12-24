cmake_minimum_required(VERSION 3.5)
project(tensorflow_cpp)
 
set(CMAKE_CXX_STANDARD 17)
 
#find_package(OpenCV 4.5.4 QUIET)
#if(NOT OpenCV_FOUND)
#   find_package(OpenCV 2.4.3 QUIET)
#   if(NOT OpenCV_FOUND)
#       message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
#   endif()
#endif()
 
set(TENSORFLOW_INCLUDES
        /usr/local/include/google
        /usr/local/include/tf
        /usr/local/include/tf/tensorflow/
        /usr/local/include/tf/third_party)
 
set(TENSORFLOW_LIBS
        /usr/local/lib/libtensorflow_cc.so.2
        /usr/local/lib/libtensorflow_framework.so.2)
 
 
include_directories(
        ${TENSORFLOW_INCLUDES}
        ${PROJECT_SOURCE_DIR}/third_party/eigen3
)
 
# add_executable(load_simple_net simple/load_simple_net.cpp)
# target_link_libraries(load_simple_net
#         ${TENSORFLOW_LIBS}
#         ${OpenCV_LIBS}
#         )

add_executable(tensorflow_cpp main.cpp)
target_link_libraries(tensorflow_cpp
        ${TENSORFLOW_LIBS}
        )
