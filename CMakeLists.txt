cmake_minimum_required(VERSION 3.5)
project(TensorStream LANGUAGES CXX CUDA)

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

#NVTX includes
if (WIN32)
    set(NVTOOLSEXT_PATH $ENV{NVTOOLSEXT_PATH})
    if (${NVTOOLSEXT_PATH})
        strip_quotes_slash(NVTOOLSEXT_PATH)
    endif()
    message(STATUS ${NVTOOLSEXT_PATH})
    if (NOT "${NVTOOLSEXT_PATH}/include" STREQUAL "/include")
        include_directories(${NVTOOLSEXT_PATH}/include)
        message(STATUS "NVTX headers found ${NVTOOLSEXT_PATH}/include")
    else()
        MESSAGE(SEND_ERROR "Can't find NVTX headers")
    endif()
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

if (UNIX)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
###############################
if(WIN32)
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

add_library(${PROJECT_NAME} SHARED ${APP_HEADERS} ${APP_SOURCE})

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
#Copy external includes to build folder
if (WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/include)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_directory ${FFMPEG_PATH}/include ${CMAKE_BINARY_DIR}/include
        COMMENT "Copying external includes")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_directory ${NVTOOLSEXT_PATH}/include ${CMAKE_BINARY_DIR}/include
        COMMENT "Copying external includes")
endif()

###############################
#FFmpeg libraries
if (UNIX)
    find_library(FFMPEG_AVCODEC avcodec)
    find_library(FFMPEG_AVUTIL avutil)
    find_library(FFMPEG_AVFORMAT avformat)
else()
    find_library(FFMPEG_AVCODEC avcodec ${FFMPEG_PATH}/lib ${FFMPEG_PATH}/bin)
    FILE(GLOB_RECURSE FFMPEG_AVCODEC_DLL ${FFMPEG_PATH}/bin/avcodec*.dll)
    find_library(FFMPEG_AVUTIL avutil ${FFMPEG_PATH}/lib ${FFMPEG_PATH}/bin)
    FILE(GLOB_RECURSE FFMPEG_AVUTIL_DLL ${FFMPEG_PATH}/bin/avutil*.dll)
    find_library(FFMPEG_AVFORMAT avformat ${FFMPEG_PATH}/lib ${FFMPEG_PATH}/bin)
    FILE(GLOB_RECURSE FFMPEG_AVFORMAT_DLL ${FFMPEG_PATH}/bin/avformat*.dll)
    FILE(GLOB_RECURSE SWRESAMPLE_DLL ${FFMPEG_PATH}/bin/swresample*.dll)

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${FFMPEG_AVCODEC_DLL} "${CMAKE_BINARY_DIR}/$(Configuration)/"
        COMMENT "Copying dependent DLL")
    
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${FFMPEG_AVFORMAT_DLL} "${CMAKE_BINARY_DIR}/$(Configuration)/"
        COMMENT "Copying dependent DLL")
    
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${FFMPEG_AVUTIL_DLL} "${CMAKE_BINARY_DIR}/$(Configuration)/"
        COMMENT "Copying dependent DLL")

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${SWRESAMPLE_DLL} "${CMAKE_BINARY_DIR}/$(Configuration)/"
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

#NVTX static libraries
if (WIN32)
    if (NOT "${NVTOOLSEXT_PATH}/lib/x64" STREQUAL "/lib/x64")
        FILE(GLOB_RECURSE NVTX_LIBS ${NVTOOLSEXT_PATH}/lib/x64/*.lib)
        target_link_libraries(${PROJECT_NAME} ${NVTX_LIBS})
        message(STATUS "NVTX static libraries found ${NVTX_LIBS}")
    else()
        MESSAGE(SEND_ERROR "Can't find NVTX static libraries")
    endif()
endif()
###############################

#NVTX dynamic libraries
if (WIN32)
    if (NOT "${NVTOOLSEXT_PATH}/bin/x64" STREQUAL "/bin/x64")
        FILE(GLOB_RECURSE NVTX_DLLS "${NVTOOLSEXT_PATH}/bin/x64/*.dll")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${NVTX_DLLS} "${CMAKE_BINARY_DIR}/$(Configuration)/"
            COMMENT "Copying dependent DLL")
        message(STATUS "NVTX dynamic libraries found ${NVTX_DLLS}")
    else()
        MESSAGE(SEND_ERROR "Can't find NVTX dynamic libraries")
    endif()
endif()
###############################