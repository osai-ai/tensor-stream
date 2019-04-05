# FindTensorStream
# -------
#
# Finds the TensorStream library
#
# This will define the following variables:
#
#   TensorStream_FOUND        -- True if the system has the TensorStream library
#   TensorStream_INCLUDE_DIRS -- The include directories for TensorStream
#   TensorStream_LIBRARIES    -- Libraries to link against
#	TensorStream_DLL_PATH     -- DLLs required by TensorStream, valid on Windows platform only
#   TensorStream_CXX_FLAGS    -- Additional (required) compiler flags
# and the following imported targets:
#
#   TensorStream

include(FindPackageHandleStandardArgs)

if(DEFINED ENV{TensorStream_INSTALL_PREFIX})
	message(STATUS "ENVIRONMENT")
  set(TensorStream_INSTALL_PREFIX $ENV{TensorStream_INSTALL_PREFIX})
else()
  # Assume we are in <install-prefix>/cmake/TensorStreamConfig.cmake
  get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
  get_filename_component(TensorStream_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../" ABSOLUTE)
endif()

enable_language(CUDA)
# Include directories.
set(TensorStream_INCLUDE_DIRS ${TensorStream_INSTALL_PREFIX}/include ${TensorStream_INSTALL_PREFIX}/include/Wrappers ${TensorStream_INSTALL_PREFIX}/build/include)
if (UNIX)
    set(TensorStream_INCLUDE_DIRS ${TensorStream_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
# TensorStream library only
find_library(TensorStream_LIBRARY TensorStream PATHS ${TensorStream_INSTALL_PREFIX} PATH_SUFFIXES build)
add_library(TensorStream UNKNOWN IMPORTED)

if (UNIX)
    find_library(CUDA_COMMON cuda PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(CUDA_COMMON_RT cudart PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    set(TensorStream_LIBRARIES ${TensorStream_LIBRARIES} ${TensorStream_LIBRARY} ${CUDA_COMMON} ${CUDA_COMMON_RT})
else()
    set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES cuda.lib cudart.lib)
    set(TensorStream_LIBRARIES ${TensorStream_LIBRARIES} ${TensorStream_LIBRARY} ${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES})
endif()

set_target_properties(TensorStream PROPERTIES
    IMPORTED_LOCATION "${TensorStream_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TensorStream_INCLUDE_DIRS}"
    CXX_STANDARD 11
)

if (WIN32)
	set(TensorStream_DLL_PATH ${TensorStream_INSTALL_PREFIX}/build)
endif()

find_package_handle_standard_args(TensorStream DEFAULT_MSG TensorStream_LIBRARIES TensorStream_INCLUDE_DIRS)
