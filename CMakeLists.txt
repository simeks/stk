cmake_minimum_required(VERSION 3.8)

message(CMAKE_BINARY_DIR " ${CMAKE_BINARY_DIR}")

option(STK_BUILD_TESTS "Build unit tests" ON)
option(STK_ENABLE_CUDA "Enables CUDA support" OFF)
option(STK_WARNINGS_ARE_ERRORS "Warnings are treated as errors" OFF)
option(STK_BUILD_WITH_DEBUG_INFO "Includes debug info in release builds" OFF)
option(STK_ENABLE_FAST_MATH "Enables fast math" OFF)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting default build type: 'Release'")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

project(stk)

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if (MSVC)
    add_definitions( -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_DEPRECATE )
    set(extra_flags "/FS")

    string(REGEX REPLACE "/W[1-3]" "/W4" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REGEX REPLACE "/W[1-3]" "/W4" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    
    if (STK_WARNINGS_ARE_ERRORS)
        set(extra_flags "${extra_flags} /WX")
    endif()

    if (STK_BUILD_WITH_DEBUG_INFO)
        set(extra_flags_release "/Zi")
    endif()

    if (STK_ENABLE_FAST_MATH)
        set(extra_flags "${extra_flags} /fp:fast")
    endif()
elseif(CMAKE_CXX_COMPILER_ID matches "GNU")
    set(extra_flags "-Wall")

    if (STK_WARNINGS_ARE_ERRORS)
        set(extra_flags "${extra_flags} -Werror")
    endif()

    if (STK_BUILD_WITH_DEBUG_INFO)
        set(extra_flags_release "-g1")
    endif()

    if (STK_ENABLE_FAST_MATH)
        set(extra_flags "${extra_flags} -ffast-math")
    endif()
elseif(CMAKE_CXX_COMPILER_ID matches "Clang")
    set(extra_flags "-Wall")

    if (STK_WARNINGS_ARE_ERRORS)
        set(extra_flags "${extra_flags} -Werror")
    endif()

    if (STK_BUILD_WITH_DEBUG_INFO)
        set(extra_flags_release "-g1")
    endif()

    if (STK_ENABLE_FAST_MATH)
        set(extra_flags "${extra_flags} -ffast-math")
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${extra_flags}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${extra_flags}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${extra_flags}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${extra_flags}")

# Catch: Test utility
set(CATCH_INCLUDE_DIR ${THIRD_PARTY_DIR}/Catch2)

add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

add_subdirectory(src)

if (STK_BUILD_TESTS)
    add_subdirectory(test)
endif()

