cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(opencv1)
SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -pthread")

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

link_directories(${OPENGL_INCLUDE_DIR}  ${GLUT_INCLUDE_DIRS})

add_executable(opencv1 opencv1.cpp)
target_link_libraries (opencv1 ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
