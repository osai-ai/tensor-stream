cmake_minimum_required(VERSION 3.7)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package (PythonLibs REQUIRED)
MESSAGE(INFO ${PYTHONLIBS_FOUND})

#affects all projects, add folder "include" to include section in VS
include_directories("include")

FILE(GLOB_RECURSE APP_SOURCE "src/*.c*")
source_group("src" FILES ${APP_SOURCE})

FILE(GLOB_RECURSE APP_HEADERS "include/*.h*")
source_group("include" FILES ${APP_HEADERS})

#include Python
include_directories(${PYTHON_INCLUDE_DIRS})

#include NVTX for profiler
include_directories($ENV{NVTOOLSEXT_PATH}/include)

#let's find dependencies
include_directories("external/ffmpeg/include")

#pytorch include for windows
set (PYTORCH_PATH "external/pytorch/build/lib.win-amd64-3.6/torch")
include_directories(${PYTORCH_PATH}/lib/include)
include_directories(${PYTORCH_PATH}/lib/include/ATen)
include_directories(${PYTORCH_PATH}/lib/include/pybind11)
include_directories(${PYTORCH_PATH}/lib/include/TH)
include_directories(${PYTORCH_PATH}/lib/include/THC)
include_directories(${PYTORCH_PATH}/lib/include/THCUNN)
include_directories(${PYTORCH_PATH}/lib/include/torch)

#include headers to project (so they will be shown in include folder)
add_executable(VideoReader ${APP_HEADERS} ${APP_SOURCE})

enable_language(CUDA)
set_target_properties(VideoReader PROPERTIES
                            CUDA_SEPERABLE_COMPILATION ON)
set (CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES cuda.lib)
MESSAGE(INFO ${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES})

FILE(GLOB_RECURSE FFMPEG_LIBS "external/ffmpeg/bin/*.lib")
target_link_libraries(VideoReader ${FFMPEG_LIBS})

target_link_libraries(VideoReader ${PYTHON_LIBRARIES})

FILE(GLOB_RECURSE NVTX_LIBS $ENV{NVTOOLSEXT_PATH}/lib/x64/*.lib)
target_link_libraries(VideoReader ${NVTX_LIBS})

FILE(GLOB_RECURSE FFMPEG_DLLS "external/ffmpeg/bin/*.dll")
file(COPY ${FFMPEG_DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)

FILE(GLOB_RECURSE PYTORCH_LIBS "${PYTORCH_PATH}/lib/*.lib")
target_link_libraries(VideoReader ${PYTORCH_LIBS})

FILE(GLOB_RECURSE PYTORCH_DLLS "${PYTORCH_PATH}/lib/*.dll")
file(COPY ${PYTORCH_DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)

FILE(GLOB_RECURSE PYTORCH_PYTHON_DLLS "${PYTORCH_PATH}/*.pyd")
file(COPY ${PYTORCH_PYTHON_DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)

FILE(GLOB_RECURSE NVTX_DLLS "$ENV{NVTOOLSEXT_PATH}/bin/x64/*.dll")
file(COPY ${NVTX_DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)