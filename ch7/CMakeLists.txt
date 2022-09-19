cmake_minimum_required(VERSION 2.8)
project(vo1)

set(CMAKE_BUILD_TYPE "Release")
# Adds definitions to the compiler command line for targets in the current directory, whether added before or after this
# command is invoked, and for the ones in sub-directories added after. This command can be used to add any flags, but it 
# is intended to add preprocessor definitions.

# "To accelerate, some complicated calculations, such as arctan, sin, and cos, are worked around by the principle of 
# trigonometric functions. In the BfMatch function, we also use the _mm_popcnt_u32 function in the SSE instruction set 
# to calculate the number of 1s in an unsigned int, which is used to achieve the effect of calculating the Hamming 
# distance."
add_definitions("-DENABLE_SSE") # Streaming SIMD Extensions (SSE) 
set(CMAKE_CXX_FLAGS "-std=c++14 -O2 ${SSE_FLAGS} -msse4")

# The list subcommands APPEND, INSERT, FILTER, PREPEND, POP_BACK, POP_FRONT, REMOVE_AT, REMOVE_ITEM, REMOVE_DUPLICATES, 
# REVERSE and SORT may create new values for the list within the current CMake variable scope
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)
find_package(G2O REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    "/usr/include/eigen3/"
)

add_executable(orb_cv orb_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})

add_executable(orb_self orb_self.cpp)
target_link_libraries(orb_self ${OpenCV_LIBS})

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d g2o_core g2o_stuff ${OpenCV_LIBS})

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d g2o_core g2o_stuff ${OpenCV_LIBS})