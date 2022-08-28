project(imageBasics)

add_executable(imageBasics imageBasics.cpp)
target_link_libraries(imageBasics ${OpenCV_LIBS})

add_executable(undistortedImage undistortedImage.cpp)
target_link_libraries(undistortedImage ${OpenCV_LIBS})
