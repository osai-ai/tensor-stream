cmake_minimum_required(VERSION 3.26)
project(TensorStream LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

function(strip_quotes_slash name)
    string(REGEX REPLACE "\\\\" "/" ${name} ${${name}})
    string(REGEX REPLACE "\"$" "" ${name} ${${name}})
    string(REGEX REPLACE "^\"" "" ${name} ${${name}})
    string(REGEX REPLACE "/$" ""  ${name} ${${name}})
    set(${name} ${${name}} PARENT_SCOPE)
endfunction()

#lib file will be located outside of Debug/Release folders
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

find_package(CUDAToolkit REQUIRED)

#affects all projects, add folder "include" to include section in VS
include_directories("include")
include_directories("include/Wrappers")

FILE(GLOB APP_SOURCE "src/*.c*")
FILE(GLOB WRAPPER_SOURCE src/Wrappers/WrapperC.c* src/Wrappers/WrapperCommon.c*)
list(APPEND APP_SOURCE ${WRAPPER_SOURCE})
source_group("src" FILES ${APP_SOURCE})

FILE(GLOB APP_HEADERS "include/*.h*")
FILE(GLOB WRAPPER_HEADER include/Wrappers/WrapperC.h* include/Wrappers/WrapperCommon.h*)
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

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
###############################
if(WIN32)
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

add_library(${PROJECT_NAME} SHARED ${APP_HEADERS} ${APP_SOURCE})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "52;60;61;70;75;80;86")

#CUDA libraries
if (WIN32)
    set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES cuda.lib)
    target_link_libraries(${PROJECT_NAME} ${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES})
else()
    find_library(CUDA_COMMON cuda PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(CUDA_COMMON_RT cudart PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    target_link_libraries(${PROJECT_NAME} ${CUDA_COMMON} ${CUDA_COMMON_RT})
endif()

###############################
#FFmpeg libraries
if (UNIX)
    find_library(FFMPEG_AVCODEC avcodec)
    find_library(FFMPEG_AVUTIL avutil)
    find_library(FFMPEG_AVFORMAT avformat)
else()
    find_library(FFMPEG_AVCODEC avcodec ${FFMPEG_PATH}/lib ${FFMPEG_PATH}/bin)
    find_library(FFMPEG_AVUTIL avutil ${FFMPEG_PATH}/lib ${FFMPEG_PATH}/bin)
    find_library(FFMPEG_AVFORMAT avformat ${FFMPEG_PATH}/lib ${FFMPEG_PATH}/bin)
    FILE(GLOB_RECURSE FFMPEG_DLL ${FFMPEG_PATH}/bin/*.dll)

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${FFMPEG_DLL} "${CMAKE_BINARY_DIR}/$(Configuration)/"
        COMMENT "Copying dependent DLL")
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