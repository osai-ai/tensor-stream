cmake_minimum_required(VERSION 3.7)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#affects all projects, add folder "include" to include section in VS
include_directories("include")

FILE(GLOB_RECURSE APP_SOURCE "src/*.c*")
source_group("src" FILES ${APP_SOURCE})

FILE(GLOB_RECURSE APP_HEADERS "include/*.h*")
source_group("include" FILES ${APP_HEADERS})

#let's find dependencies
include_directories("external/ffmpeg/include")

#include headers to project (so they will be shown in include folder)
add_executable(VideoReader ${APP_HEADERS} ${APP_SOURCE})

enable_language(CUDA)
set_target_properties(VideoReader PROPERTIES
                            CUDA_SEPERABLE_COMPILATION ON)

FILE(GLOB_RECURSE FFMPEG_LIBS "external/ffmpeg/bin/*.lib")
target_link_libraries(VideoReader ${FFMPEG_LIBS})

FILE(GLOB_RECURSE FFMPEG_DLLS "external/ffmpeg/bin/*.dll")
file(COPY ${FFMPEG_DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)