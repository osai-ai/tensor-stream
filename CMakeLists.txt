cmake_minimum_required(VERSION 3.5)

function(strip_quotes_slash name)
    string(REGEX REPLACE "\\\\" "/" ${name} ${${name}})
    string(REGEX REPLACE "\"$" "" ${name} ${${name}})
    string(REGEX REPLACE "^\"" "" ${name} ${${name}})
    string(REGEX REPLACE "/$" ""  ${name} ${${name}})
    set(${name} ${${name}} PARENT_SCOPE)
endfunction()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")

#affects all projects, add folder "include" to include section in VS
include_directories("include")

FILE(GLOB_RECURSE APP_SOURCE "src/*.c*")
source_group("src" FILES ${APP_SOURCE})

FILE(GLOB_RECURSE APP_HEADERS "include/*.h*")
source_group("include" FILES ${APP_HEADERS})

#Python includes
find_package (PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
message(STATUS "Python headers found ${PYTHON_INCLUDE_DIRS}")
###############################
#FFmpeg includes
if (WIN32)
    set(FFMPEG_PATH $ENV{FFMPEG_PATH})
    if (NOT ${FFMPEG_PATH} STREQUAL "")
        strip_quotes_slash(FFMPEG_PATH)
    endif()
    if (NOT "${FFMPEG_PATH}/include" STREQUAL "/include")
        include_directories(${FFMPEG_PATH}/include)
        message(STATUS "FFmpeg headers found ${FFMPEG_PATH}/include")
    else()
        MESSAGE(FATAL_ERROR "Can't find FFmpeg headers. Please set FFmpeg root folder path to FFMPEG_PATH variable")
    endif()
endif()


if (WIN32)
    set(TORCH_INSTALL_PREFIX $ENV{PYTORCH_PATH})
    if (NOT ${TORCH_INSTALL_PREFIX} STREQUAL "")
        strip_quotes_slash(TORCH_INSTALL_PREFIX)
    endif()
else()
    find_package(Torch REQUIRED)
endif()

if (EXISTS ${TORCH_INSTALL_PREFIX}/lib/include)
    include_directories(${TORCH_INSTALL_PREFIX}/lib/include)
    include_directories(${TORCH_INSTALL_PREFIX}/lib/include/torch/csrc/api/include)
    include_directories(${TORCH_INSTALL_PREFIX}/lib/include/ATen)
    include_directories(${TORCH_INSTALL_PREFIX}/lib/include/pybind11)
    include_directories(${TORCH_INSTALL_PREFIX}/lib/include/TH)
    include_directories(${TORCH_INSTALL_PREFIX}/lib/include/THC)
    include_directories(${TORCH_INSTALL_PREFIX}/lib/include/THCUNN)
    include_directories(${TORCH_INSTALL_PREFIX}/lib/include/torch)
else()
    include_directories(${TORCH_INSTALL_PREFIX}/include/ATen)
    include_directories(${TORCH_INSTALL_PREFIX}/include/pybind11)
    include_directories(${TORCH_INSTALL_PREFIX}/include/TH)
    include_directories(${TORCH_INSTALL_PREFIX}/include/THC)
    include_directories(${TORCH_INSTALL_PREFIX}/include/THCUNN)
    include_directories(${TORCH_INSTALL_PREFIX}/include/torch)
endif()

enable_language(CUDA)
if (LINUX)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
###############################

#include headers to project (so they will be shown in include folder)
add_executable(VideoReader ${APP_HEADERS} ${APP_SOURCE})

#Python libraries
target_link_libraries(VideoReader ${PYTHON_LIBRARIES})
message(STATUS "Python static libraries found ${PYTHON_LIBRARIES}")
###############################

#CUDA libraries
if (WIN32)
    set_target_properties(VideoReader PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
    #do we need cuda.lib here?
    set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES cuda.lib)
else()
    find_library(CUDA_COMMON cuda PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(CUDA_COMMON_RT cudart PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    target_link_libraries(VideoReader ${CUDA_COMMON} ${CUDA_COMMON_RT})
endif()
###############################

#FFmpeg static libraries
if (LINUX)
    find_library(FFMPEG_AVCODEC avcodec)
    find_library(FFMPEG_AVUTIL avutil)
    find_library(FFMPEG_AVFORMAT avformat)
endif()
if (WIN32)
    find_library(FFMPEG_AVCODEC avcodec ${FFMPEG_PATH}/lib ${FFMPEG_PATH}/bin)
    find_library(FFMPEG_AVUTIL avutil ${FFMPEG_PATH}/lib ${FFMPEG_PATH}/bin)
    find_library(FFMPEG_AVFORMAT avformat ${FFMPEG_PATH}/lib ${FFMPEG_PATH}/bin)
    FILE(GLOB_RECURSE FFMPEG_DLLS ${FFMPEG_PATH}/bin/*.dll)
    FILE(COPY ${FFMPEG_DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)
    set(CMAKE_MSVCIDE_RUN_PATH ${FFMPEG_DLLS})
    #set(CMAKE_MSVCIDE_RUN_PATH ${FFMPEG_DLLS})
endif()
message(STATUS "FFMPEG ${FFMPEG_AVCODEC} ${FFMPEG_AVUTIL} ${FFMPEG_AVFORMAT}")
if (NOT (${FFMPEG_AVCODEC} STREQUAL "" OR ${FFMPEG_AVUTIL} STREQUAL "" OR ${FFMPEG_AVFORMAT} STREQUAL ""))
    target_link_libraries(VideoReader ${FFMPEG_AVCODEC} ${FFMPEG_AVUTIL} ${FFMPEG_AVFORMAT})
else()
    if(WIN32)
        message(FATAL_ERROR "Add path to FFmpeg folder to FFMPEG_PATH environment variable")
    else()
        message(FATAL_ERROR "Add path to FFmpeg folder to CMAKE_PREFIX_PATH")
    endif()
endif()
###############################


if (WIN32)
    set(PYTORCH_PATH $ENV{PYTORCH_PATH})
    if (NOT ${PYTORCH_PATH} STREQUAL "")
        strip_quotes_slash(PYTORCH_PATH)
    endif()
    #static libraries
    FILE(GLOB_RECURSE PYTORCH_LIBS "${PYTORCH_PATH}/lib/*.lib")
    target_link_libraries(VideoReader ${PYTORCH_LIBS})
    FILE(GLOB_RECURSE PYTORCH_PYTHON_DLLS "${PYTORCH_PATH}/*.pyd")
    #dynamic libraries
    FILE(COPY ${PYTORCH_PYTHON_DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)
    set(CMAKE_MSVCIDE_RUN_PATH ${PYTORCH_PYTHON_DLLS})
    FILE(GLOB_RECURSE PYTORCH_DLLS "${PYTORCH_PATH}/lib/*.dll")
    FILE(COPY ${PYTORCH_DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)
    set(CMAKE_MSVCIDE_RUN_PATH ${PYTORCH_DLLS})
endif()
if (LINUX)
    #Pytorch static libraries
    foreach(PREFIX_PATH ${CMAKE_PREFIX_PATH})
        if (NOT ${PREFIX_PATH} STREQUAL "")    
            target_link_libraries(VideoReader ${TORCH_LIBRARIES})
            find_library(PYTORCH_TORCH_PYTHON torch_python)
            find_library(PYTORCH_C10 c10)
            find_library(PYTORCH_C10_GPU c10_cuda)
            target_link_libraries(VideoReader ${PYTORCH_LIBS})
            if (${PYTORCH_TORCH_PYTHON} AND ${PYTORCH_C10} AND ${PYTORCH_C10_GPU})
               target_link_libraries(VideoReader ${PYTORCH_TORCH_PYTHON} ${PYTORCH_C10} ${PYTORCH_C10_GPU})
            endif()
        endif()
    endforeach(PREFIX_PATH)
endif()
###############################

if (WIN32)
    #FFmpeg dynamic libraries
    ###############################

    #NVTX dynamic libraries
        #FILE(GLOB_RECURSE NVTX_DLLS "${NVTOOLSEXT_PATH}/bin/x64/*.dll")
        #FILE(COPY ${NVTX_DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)
        #message(STATUS "NVTX dynamic libraries found ${NVTX_DLLS}")
        #MESSAGE(SEND_ERROR "Can't find NVTX dynamic libraries")
    ###############################
endif()
