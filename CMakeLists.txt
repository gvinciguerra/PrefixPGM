cmake_minimum_required(VERSION 3.12)
project(PrefixPGMIndex)

set(CMAKE_CXX_STANDARD 17)
if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()

include_directories(wide-integer/math/wide_integer)
add_executable(example example.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(example PUBLIC OpenMP::OpenMP_CXX)
endif()