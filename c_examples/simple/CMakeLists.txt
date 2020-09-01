cmake_minimum_required(VERSION 3.5)
project(Sample LANGUAGES CXX)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
endif()

FILE(GLOB_RECURSE APP_SOURCE "src/*.c*")
source_group("src" FILES ${APP_SOURCE})

find_package(TensorStream REQUIRED)

include_directories(${TensorStream_INCLUDE_DIRS})

if (UNIX)
	find_package(SDL2 REQUIRED)
	include_directories(${SDL2_INCLUDE_DIRS})
	message(STATUS ${SDL2_INCLUDE_DIRS})
else()
	include_directories("D:\\Work\\external\\SDL\\include")
endif()

add_executable(${PROJECT_NAME} ${APP_SOURCE})
target_link_libraries(${PROJECT_NAME} ${TensorStream_LIBRARIES})
if (UNIX)
	target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES})
else()
	target_link_libraries(${PROJECT_NAME} "D:\\Work\\external\\SDL\\build\\Release\\SDL2.lib" "D:\\Work\\external\\SDL\\build\\Release\\SDL2main.lib")
endif()


if (WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_directory ${TensorStream_DLL_PATH}/$(Configuration) ${CMAKE_BINARY_DIR}/$(Configuration)
        COMMENT "Copying dependent DLL")

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy "D:\\Work\\external\\SDL\\build\\Release\\SDL2.dll" ${CMAKE_BINARY_DIR}/$(Configuration)
        COMMENT "Copying dependent DLL")
endif()
