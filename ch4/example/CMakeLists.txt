
find_package(Pangolin REQUIRED)
find_package(fmt REQUIRED)
set(FMT_LIBRARIES fmt::fmt)
include_directories(${Pangolin_INCLUDE_DIRS})
add_executable(trajectoryError trajectoryError.cpp)
target_link_libraries(trajectoryError ${Pangolin_LIBRARIES} ${FMT_LIBRARIES})