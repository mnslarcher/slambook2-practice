cmake_minimum_required(VERSION 2.8)
project(chapter3)

set(CMAKE_CXX_FLAGS "-std=c++17")

add_subdirectory(useEigen)
add_subdirectory(useGeometry)
add_subdirectory(visualizeGeometry)
add_subdirectory(examples)