cmake_minimum_required(VERSION 2.8)
project(landmark)

set(CMAKE_CXX_STANDARD 14)
file (GLOB source_files "src/*.cpp")

SET(OpenCV_DIR /usr/local/share/OpenCV/OpenCVConfig.cmake)
find_package( OpenCV REQUIRED )
include_directories( /usr/local/include include )

add_executable(landmark ${source_files})
target_link_libraries(landmark ${OpenCV_LIBS})
