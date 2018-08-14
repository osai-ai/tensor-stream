cmake_minimum_required(VERSION 3.7)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#affects all projects, add folder "include" to include section in VS
include_directories("include")

FILE(GLOB_RECURSE APP_SOURCE "src/*.c*")
source_group("src" FILES ${APP_SOURCE})

FILE(GLOB_RECURSE APP_HEADERS "include/*.h*")
source_group("include" FILES ${APP_HEADERS})

#include headers to project (so they will be shown in include folder)
add_executable(VideoReader ${APP_HEADERS} ${APP_SOURCE})