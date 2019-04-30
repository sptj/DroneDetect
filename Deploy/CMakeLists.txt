cmake_minimum_required(VERSION 3.13)
project(sptj)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(sptj sptj.cpp DetectManager.cpp get_anchors.cpp get_anchors.h DetectManager.h)
#add_library(sptj sptj.h sptj.cpp DetectManager.cpp get_anchors.cpp get_anchors.h DetectManager.h)
target_link_libraries(sptj ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES}  )
set_property(TARGET sptj PROPERTY _CXX_STANDARD 11)
