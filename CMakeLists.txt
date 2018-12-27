cmake_minimum_required(VERSION 3.5)

function(strip_quotes_slash name)
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
    file(GLOB_RECURE FFMPEG_HEADER ${CMAKE_PREFIX_PATH}/*avformat.h)
    get_filename_component(ITEM_PATH ${FFMPEG_HEADER} FFMPEG_HEADER_DIR)
    get_filename_component(PARENT_DIR ${FFMPEG_HEADER_DIR} FFMPEG_PARENT_DIR) 
    include_directories(${FFMPEG_PARENT_DIR})
    message(STATUS "FFmpeg headers found ${FFMPEG_PARENT_DIR}")
endif()

#Need point to TorchConfig.cmake in CMAKE_PREFIX_PATH
find_package(Torch REQUIRED)
message(STATUS "Torch directories ${TORCH_INCLUDE_DIRS}")
include_directories(${TORCH_INCLUDE_DIRS})
message(STATUS "Torch prefix path ${TORCH_INSTALL_PREFIX}")
if (EXISTS ${TORCH_INSTALL_PREFIX}/lib/include)
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
if (NOT ${CMAKE_PREFIX_PATH} STREQUAL "")
    find_library(FFMPEG_AVCODEC avcodec)
    message(STATUS ${FFMPEG_AVCODEC})
    find_library(FFMPEG_AVUTIL avutil)
    message(STATUS ${FFMPEG_AVUTIL})
    find_library(FFMPEG_AVFORMAT avformat)
    message(STATUS ${FFMPEG_AVFORMAT})
    if (${FFMPEG_AVCODEC} STREQUAL "" OR ${FFMPEG_AVUTIL} STREQUAL "" OR ${FFMPEG_AVFORMAT} STREQUAL "")
        message(FATAL_ERROR "Can't find ffmpeg libraries. Please set FFmpeg folder path to CMAKE_PREFIX_PATH")
    endif()
    target_link_libraries(VideoReader ${FFMPEG_AVCODEC} ${FFMPEG_AVUTIL} ${FFMPEG_AVFORMAT})
else()
    message(STATUS ${CMAKE_PREFIX_PATH})
    message(FATAL_ERROR "Add path to FFmpeg folder to CMAKE_PREFIX_PATH")
endif()
###############################

message(STATUS "Torch libraries ${TORCH_LIBRARIES}")
target_link_libraries(VideoReader ${TORCH_LIBRARIES})
#Pytorch static libraries
if (NOT ${CMAKE_PREFIX_PATH} STREQUAL "")
    if (WIN32)
        #find_library(PYTORCH_C _C ${CMAKE_PREFIX_PATH}/lib ${CMAKE_PREFIX_PATH}/bin)
    	target_link_libraries(VideoReader ${PYTORCH_LIBS})
    else()
        find_library(PYTORCH_TORCH_PYTHON torch_python)
        find_library(PYTORCH_C10 c10)
        find_library(PYTORCH_C10_GPU c10_gpu)
    	target_link_libraries(VideoReader ${PYTORCH_LIBS})
    endif()
else()
    message(FATAL_ERROR "Add path to Pytorch folder to CMAKE_PREFIX_PATH")
endif()
###############################

if (WIN32)
    #Pytorch dynamic libraries
        FILE(GLOB_RECURSE PYTORCH_PYTHON_DLLS "${PYTORCH_PATH}/*.pyd")
        FILE(COPY ${PYTORCH_PYTHON_DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)
        set(CMAKE_MSVCIDE_RUN_PATH ${PYTORCH_PYTHON_DLLS})
        FILE(GLOB_RECURSE PYTORCH_DLLS "${PYTORCH_PATH}/lib/*.dll")
        FILE(COPY ${PYTORCH_DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)
        set(CMAKE_MSVCIDE_RUN_PATH ${PYTORCH_DLLS})
        message(STATUS "Pytorch dynamic libraries found ${PYTORCH_PYTHON_DLLS} ${PYTORCH_DLLS}")
    ###############################

    #FFmpeg dynamic libraries
        FILE(GLOB_RECURSE FFMPEG_DLLS ${FFMPEG_PATH}/bin/*.dll)
        FILE(COPY ${FFMPEG_DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)
        set(CMAKE_MSVCIDE_RUN_PATH ${FFMPEG_DLLS})
        message(STATUS "FFmpeg dynamic libraries found ${FFMPEG_DLLS}")
    ###############################

    #NVTX dynamic libraries
        FILE(GLOB_RECURSE NVTX_DLLS "${NVTOOLSEXT_PATH}/bin/x64/*.dll")
        FILE(COPY ${NVTX_DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)
        message(STATUS "NVTX dynamic libraries found ${NVTX_DLLS}")
        MESSAGE(SEND_ERROR "Can't find NVTX dynamic libraries")
    ###############################
endif()
