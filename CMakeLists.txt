cmake_minimum_required(VERSION 2.8.12)
project(CV_project)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(CV_project src/CV_project.cpp src/Superpixel.cpp src/SuperpixelCreation.cpp src/Utils.cpp src/Svm.cpp)

target_link_libraries(CV_project ${OpenCV_LIBS})
