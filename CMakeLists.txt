cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(opencv1)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenCV REQUIRED)

link_directories(${OPENGL_INCLUDE_DIR}  ${GLUT_INCLUDE_DIRS})

add_executable(opencv1 opencv1.cpp)
target_link_libraries (opencv1 ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${OpenCV_LIBS})
