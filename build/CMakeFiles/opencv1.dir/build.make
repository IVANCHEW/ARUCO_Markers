# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ivan/c++/openCV1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ivan/c++/openCV1/build

# Include any dependencies generated for this target.
include CMakeFiles/opencv1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/opencv1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/opencv1.dir/flags.make

CMakeFiles/opencv1.dir/opencv1.cpp.o: CMakeFiles/opencv1.dir/flags.make
CMakeFiles/opencv1.dir/opencv1.cpp.o: ../opencv1.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ivan/c++/openCV1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/opencv1.dir/opencv1.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencv1.dir/opencv1.cpp.o -c /home/ivan/c++/openCV1/opencv1.cpp

CMakeFiles/opencv1.dir/opencv1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv1.dir/opencv1.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ivan/c++/openCV1/opencv1.cpp > CMakeFiles/opencv1.dir/opencv1.cpp.i

CMakeFiles/opencv1.dir/opencv1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv1.dir/opencv1.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ivan/c++/openCV1/opencv1.cpp -o CMakeFiles/opencv1.dir/opencv1.cpp.s

CMakeFiles/opencv1.dir/opencv1.cpp.o.requires:

.PHONY : CMakeFiles/opencv1.dir/opencv1.cpp.o.requires

CMakeFiles/opencv1.dir/opencv1.cpp.o.provides: CMakeFiles/opencv1.dir/opencv1.cpp.o.requires
	$(MAKE) -f CMakeFiles/opencv1.dir/build.make CMakeFiles/opencv1.dir/opencv1.cpp.o.provides.build
.PHONY : CMakeFiles/opencv1.dir/opencv1.cpp.o.provides

CMakeFiles/opencv1.dir/opencv1.cpp.o.provides.build: CMakeFiles/opencv1.dir/opencv1.cpp.o


# Object files for target opencv1
opencv1_OBJECTS = \
"CMakeFiles/opencv1.dir/opencv1.cpp.o"

# External object files for target opencv1
opencv1_EXTERNAL_OBJECTS =

opencv1: CMakeFiles/opencv1.dir/opencv1.cpp.o
opencv1: CMakeFiles/opencv1.dir/build.make
opencv1: /usr/lib/x86_64-linux-gnu/libGLU.so
opencv1: /usr/lib/x86_64-linux-gnu/libGL.so
opencv1: /usr/lib/x86_64-linux-gnu/libglut.so
opencv1: /usr/lib/x86_64-linux-gnu/libXmu.so
opencv1: /usr/lib/x86_64-linux-gnu/libXi.so
opencv1: /opt/ros/kinetic/lib/libopencv_xphoto3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_xobjdetect3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_tracking3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_surface_matching3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_structured_light3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_stereo3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_saliency3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_rgbd3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_reg3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_plot3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_optflow3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_line_descriptor3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_hdf3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_fuzzy3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_dpm3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_dnn3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_datasets3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_cvv3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_ccalib3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_bioinspired3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_bgsegm3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_aruco3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_viz3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_videostab3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_superres3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_stitching3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_photo3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_text3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_face3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_ximgproc3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_xfeatures2d3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_shape3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_video3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_objdetect3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_calib3d3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_features2d3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_ml3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_highgui3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_videoio3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_imgcodecs3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_imgproc3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_flann3.so.3.1.0
opencv1: /opt/ros/kinetic/lib/libopencv_core3.so.3.1.0
opencv1: CMakeFiles/opencv1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ivan/c++/openCV1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable opencv1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencv1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/opencv1.dir/build: opencv1

.PHONY : CMakeFiles/opencv1.dir/build

CMakeFiles/opencv1.dir/requires: CMakeFiles/opencv1.dir/opencv1.cpp.o.requires

.PHONY : CMakeFiles/opencv1.dir/requires

CMakeFiles/opencv1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/opencv1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/opencv1.dir/clean

CMakeFiles/opencv1.dir/depend:
	cd /home/ivan/c++/openCV1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ivan/c++/openCV1 /home/ivan/c++/openCV1 /home/ivan/c++/openCV1/build /home/ivan/c++/openCV1/build /home/ivan/c++/openCV1/build/CMakeFiles/opencv1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/opencv1.dir/depend

