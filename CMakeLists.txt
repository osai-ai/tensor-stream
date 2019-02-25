cmake_minimum_required(VERSION 3.5)
project(TensorStream)

function(strip_quotes_slash name)
    string(REGEX REPLACE "\\\\" "/" ${name} ${${name}})
    string(REGEX REPLACE "\"$" "" ${name} ${${name}})
    string(REGEX REPLACE "^\"" "" ${name} ${${name}})
    string(REGEX REPLACE "/$" ""  ${name} ${${name}})
    set(${name} ${${name}} PARENT_SCOPE)
endfunction()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")


#affects all projects, add folder "include" to include section in VS
include_directories("include")
include_directories("include/Wrappers")

FILE(GLOB APP_SOURCE "src/*.c*")
FILE(GLOB WRAPPER_SOURCE src/Wrappers/WrapperC.c*)
list(APPEND APP_SOURCE ${WRAPPER_SOURCE})
source_group("src" FILES ${APP_SOURCE})

FILE(GLOB APP_HEADERS "include/*.h*")
FILE(GLOB WRAPPER_HEADER include/Wrappers/WrapperC.h*)
list(APPEND APP_HEADERS ${WRAPPER_HEADER})
source_group("include" FILES ${APP_HEADERS})

#FFmpeg includes
if (WIN32)
    set(FFMPEG_PATH $ENV{FFMPEG_PATH})
    if (NOT ${FFMPEG_PATH} STREQUAL "")
        strip_quotes_slash(FFMPEG_PATH)
    else()
        message(FATAL_ERROR "Set path to FFmpeg to FFMPEG_PATH environment variable")
    endif()
    if (NOT "${FFMPEG_PATH}/include" STREQUAL "/include")
        include_directories(${FFMPEG_PATH}/include)
        message(STATUS "FFmpeg headers found ${FFMPEG_PATH}/include")
    else()
        MESSAGE(FATAL_ERROR "Can't find FFmpeg headers. Please set FFmpeg root folder path to FFMPEG_PATH variable")
    endif()
endif()

enable_language(CUDA)
if (UNIX)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
###############################

#include headers to project (so they will be shown in include folder)

if(WIN32)
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

add_library(${PROJECT_NAME} SHARED ${APP_HEADERS} ${APP_SOURCE})

#CUDA libraries
if (WIN32)
    set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES cuda.lib)
else()
    find_library(CUDA_COMMON cuda PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(CUDA_COMMON_RT cudart PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    target_link_libraries(${PROJECT_NAME} ${CUDA_COMMON} ${CUDA_COMMON_RT})
endif()
###############################

#FFmpeg static libraries
if (UNIX)
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
endif()
if (FFMPEG_AVCODEC AND FFMPEG_AVUTIL AND FFMPEG_AVFORMAT)
    target_link_libraries(${PROJECT_NAME} ${FFMPEG_AVCODEC} ${FFMPEG_AVUTIL} ${FFMPEG_AVFORMAT})
else()
    if(WIN32)
        message(FATAL_ERROR "Add path to FFmpeg folder to FFMPEG_PATH environment variable")
    else()
        message(FATAL_ERROR "Add path to FFmpeg folder to CMAKE_PREFIX_PATH")
    endif()
endif()
###############################