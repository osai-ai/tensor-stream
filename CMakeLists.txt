cmake_minimum_required(VERSION 3.7)

function(strip_quotes_slash name)
	string(REGEX REPLACE "\"$" "" ${name} ${${name}})
	string(REGEX REPLACE "^\"" "" ${name} ${${name}})
	string(REGEX REPLACE "/$" ""  ${name} ${${name}})
	set(${name} ${${name}} PARENT_SCOPE)
endfunction()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

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

#NVTX includes
set(NVTOOLSEXT_PATH $ENV{NVTOOLSEXT_PATH})
strip_quotes_slash(NVTOOLSEXT_PATH)
if (NOT "${NVTOOLSEXT_PATH}/include" STREQUAL "/include")
	include_directories(${NVTOOLSEXT_PATH}/include)
	message(STATUS "NVTX headers found ${NVTOOLSEXT_PATH}/include")
else()
	MESSAGE(SEND_ERROR "Can't find NVTX headers")
endif()
###############################

#FFmpeg includes
set(FFMPEG_PATH $ENV{FFMPEG_PATH})
strip_quotes_slash(FFMPEG_PATH)
if (NOT "${FFMPEG_PATH}/include" STREQUAL "/include")
	include_directories(${FFMPEG_PATH}/include)
	message(STATUS "FFmpeg headers found ${FFMPEG_PATH}/include")
else()
	MESSAGE(FATAL_ERROR "Can't find FFmpeg headers")
endif()
###############################

#pytorch include for windows
set(PYTORCH_PATH $ENV{PYTORCH_PATH})
strip_quotes_slash(PYTORCH_PATH)
if (NOT "${PYTORCH_PATH}/lib/include" STREQUAL "/lib/include")
	list(APPEND PYTORCH_INCLUDES ${PYTORCH_PATH}/lib/include)
	list(APPEND PYTORCH_INCLUDES ${PYTORCH_PATH}/lib/include/ATen)
	list(APPEND PYTORCH_INCLUDES ${PYTORCH_PATH}/lib/include/pybind11)
	list(APPEND PYTORCH_INCLUDES ${PYTORCH_PATH}/lib/include/TH)
	list(APPEND PYTORCH_INCLUDES ${PYTORCH_PATH}/lib/include/THC)
	list(APPEND PYTORCH_INCLUDES ${PYTORCH_PATH}/lib/include/THCUNN)
	list(APPEND PYTORCH_INCLUDES ${PYTORCH_PATH}/lib/include/torch)
	include_directories(${PYTORCH_INCLUDES})
	MESSAGE(STATUS "Pytorch headers found ${PYTORCH_INCLUDES}")
else()
	MESSAGE(FATAL_ERROR "Can't find Pytorch headers")
endif()
###############################

#include headers to project (so they will be shown in include folder)
add_executable(VideoReader ${APP_HEADERS} ${APP_SOURCE})

#link CUDA
enable_language(CUDA)
set_target_properties(VideoReader PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES cuda.lib)
###############################

#Python libraries
target_link_libraries(VideoReader ${PYTHON_LIBRARIES})
message(STATUS "Python static libraries found ${PYTHON_LIBRARIES}")
###############################

#FFmpeg static libraries
if (NOT "${FFMPEG_PATH}/bin" STREQUAL "/bin")
    FILE(GLOB_RECURSE FFMPEG_LIBS ${FFMPEG_PATH}/bin/*.lib)
	target_link_libraries(VideoReader ${FFMPEG_LIBS})
	message(STATUS "FFmpeg static libraries found ${FFMPEG_LIBS}")
else()
	MESSAGE(FATAL_ERROR "Can't find FFmpeg static libraries")
endif()
###############################

#NVTX static libraries
if (NOT "${NVTOOLSEXT_PATH}/lib/x64" STREQUAL "/lib/x64")
	FILE(GLOB_RECURSE NVTX_LIBS ${NVTOOLSEXT_PATH}/lib/x64/*.lib)
	target_link_libraries(VideoReader ${NVTX_LIBS})
	message(STATUS "NVTX static libraries found ${NVTX_LIBS}")
else()
	MESSAGE(SEND_ERROR "Can't find NVTX static libraries")
endif()
###############################

#Pytorch static libraries
if (NOT "${PYTORCH_PATH}/lib" STREQUAL "/lib")
	FILE(GLOB_RECURSE PYTORCH_LIBS "${PYTORCH_PATH}/lib/*.lib")
	target_link_libraries(VideoReader ${PYTORCH_LIBS})
	message(STATUS "Pytorch static libraries found ${PYTORCH_LIBS}")
else()
	MESSAGE(FATAL_ERROR "Can't find Pytorch static libraries")
endif()
###############################

#Pytorch dynamic libraries
if (NOT "${PYTORCH_PATH}/lib" STREQUAL "/lib" AND NOT "${PYTORCH_PATH}" STREQUAL "")
	FILE(GLOB_RECURSE PYTORCH_PYTHON_DLLS "${PYTORCH_PATH}/*.pyd")
	FILE(COPY ${PYTORCH_PYTHON_DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)
	set(CMAKE_MSVCIDE_RUN_PATH ${PYTORCH_PYTHON_DLLS})
	FILE(GLOB_RECURSE PYTORCH_DLLS "${PYTORCH_PATH}/lib/*.dll")
	FILE(COPY ${PYTORCH_DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)
	set(CMAKE_MSVCIDE_RUN_PATH ${PYTORCH_DLLS})
	message(STATUS "Pytorch dynamic libraries found ${PYTORCH_PYTHON_DLLS} ${PYTORCH_DLLS}")
else()
	MESSAGE(FATAL_ERROR "Can't find Pytorch dynamic libraries")
endif()
###############################

#FFmpeg dynamic libraries
if (NOT "${FFMPEG_PATH}/bin" STREQUAL "/bin")
	FILE(GLOB_RECURSE FFMPEG_DLLS ${FFMPEG_PATH}/bin/*.dll)
	FILE(COPY ${FFMPEG_DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)
	set(CMAKE_MSVCIDE_RUN_PATH ${FFMPEG_DLLS})
	message(STATUS "FFmpeg dynamic libraries found ${FFMPEG_DLLS}")
else()
	MESSAGE(FATAL_ERROR "Can't find FFmpeg dynamic libraries")
endif()
###############################

#NVTX dynamic libraries
if (NOT "${NVTOOLSEXT_PATH}/bin/x64" STREQUAL "/bin/x64")
	FILE(GLOB_RECURSE NVTX_DLLS "${NVTOOLSEXT_PATH}/bin/x64/*.dll")
	FILE(COPY ${NVTX_DLLS} DESTINATION ${PROJECT_SOURCE_DIR}/build)
	message(STATUS "NVTX dynamic libraries found ${NVTX_DLLS}")
else()
	MESSAGE(SEND_ERROR "Can't find NVTX dynamic libraries")
endif()
###############################