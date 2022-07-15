cmake_minimum_required(VERSION 2.8)

project(HelloSLAM)

set(CMAKE_BUILD_TYPE "Debug")

add_executable(helloSLAM helloSLAM.cpp)

# library: collection of pre-compiled code usable by other programs

# STATIC libraries are archives of object files for use when linking other targets
# Linked at compile time
# The executable keeps its own copy of the library
add_library(hello libHelloSLAM.cpp)

# SHARED libraries are linked dynamically and loaded at runtime
# Contains code to be shared by multiple programs
# Loaded at runtime
add_library(hello_shared SHARED libHelloSLAM.cpp)

# Dynamic libraries are loaded into memory only once, whereas for static libraries each executable must load the library
# into memory

# Dynamic
# + Avoids having to maintain multiple copies of the same library
# + Most of the time, dynamic library upgrade does not require recompiling other part
# - Must be located appropriately to make sure the executable can find libraries at runtime

# Static:
# + Monolithic package
# - Increase the size of the application
# - Modifying/upgrading the library code requires rerun compiling/linking of other parts of the application


add_executable(useHello useHello.cpp)
target_link_libraries(useHello hello_shared)