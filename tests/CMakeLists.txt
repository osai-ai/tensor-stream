cmake_minimum_required(VERSION 3.5)
project(Tests LANGUAGES CXX CUDA)

function(strip_quotes_slash name)
    string(REGEX REPLACE "\\\\" "/" ${name} ${${name}})
    string(REGEX REPLACE "\"$" "" ${name} ${${name}})
    string(REGEX REPLACE "^\"" "" ${name} ${${name}})
    string(REGEX REPLACE "/$" ""  ${name} ${${name}})
    set(${name} ${${name}} PARENT_SCOPE)
endfunction()

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download" )
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download" )

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This adds
# the following targets: gtest, gtest_main, gmock
# and gmock_main
add_subdirectory("${CMAKE_BINARY_DIR}/googletest-src"
                 "${CMAKE_BINARY_DIR}/googletest-build")

# The gtest/gmock targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if(CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include"
                        "${gmock_SOURCE_DIR}/include")
endif()

FILE(GLOB_RECURSE APP_SOURCE "src/*.c*")
source_group("src" FILES ${APP_SOURCE})

include_directories("${PROJECT_SOURCE_DIR}/../include")
include_directories("${PROJECT_SOURCE_DIR}/../include/Wrappers")
#include headers to project (so they will be shown in include folder)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
message(STATUS "GTEST SOURCE DIR ${gtest_SOURCE_DIR}")

find_package(TensorStream REQUIRED)
include_directories(${TensorStream_INCLUDE_DIRS})
###############################
add_executable(${PROJECT_NAME} ${APP_SOURCE})

#CUDA libraries
if (WIN32)
    set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES cuda.lib cudart.lib)
    target_link_libraries(${PROJECT_NAME} ${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES})
else()
    find_library(CUDA_COMMON cuda PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(CUDA_COMMON_RT cudart PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    target_link_libraries(${PROJECT_NAME} ${CUDA_COMMON} ${CUDA_COMMON_RT})
endif()


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

#FFmpeg libraries
if (UNIX)
    find_library(FFMPEG_AVCODEC avcodec)
    find_library(FFMPEG_AVUTIL avutil)
    find_library(FFMPEG_AVFORMAT avformat)
else()
	find_library(FFMPEG_AVCODEC avcodec ${FFMPEG_PATH}/lib ${FFMPEG_PATH}/bin)
    find_library(FFMPEG_AVUTIL avutil ${FFMPEG_PATH}/lib ${FFMPEG_PATH}/bin)
    find_library(FFMPEG_AVFORMAT avformat ${FFMPEG_PATH}/lib ${FFMPEG_PATH}/bin)
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

if (WIN32)
    target_link_libraries(${PROJECT_NAME} ${TensorStream_LIBRARIES})
else()
    target_link_libraries(${PROJECT_NAME} ${TensorStream_LIBRARIES})
endif()

#FFmpeg static libraries
###############################
if (WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_directory ${TensorStream_DLL_PATH}/$(Configuration) ${CMAKE_BINARY_DIR}/$(Configuration)
        COMMENT "Copying dependent DLL")
else()
    target_link_libraries(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/../build/libTensorStream.so)
endif()

target_link_libraries(${PROJECT_NAME} gtest gtest_main)
add_test( ${PROJECT_NAME} ${PROJECT_NAME})