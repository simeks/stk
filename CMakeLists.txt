cmake_minimum_required(VERSION 3.8)

option(STK_BUILD_EXAMPLES "Build examples" ON)
option(STK_BUILD_TESTS "Build unit tests" ON)
option(STK_USE_CUDA "Enables CUDA support" OFF)
option(STK_WARNINGS_ARE_ERRORS "Warnings are treated as errors" OFF)
option(STK_BUILD_WITH_DEBUG_INFO "Includes debug info in release builds" OFF)
option(STK_ENABLE_FAST_MATH "Enables fast math" OFF)
option(STK_LOGGING_PREFIX_FILE "Add file name as prefix to each log message" OFF)
option(STK_ITK_BRIDGE "Bridge library to interoperate with ITK" OFF)
option(STK_STACK_TRACE "Print a stack trace on errors" OFF)

# The stack trace is of little use without symbols
if(STK_STACK_TRACE)
	set(STK_BUILD_WITH_DEBUG_INFO ON)
endif()

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting default build type: 'Release'")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

project(stk)

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
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
    set(EXTRA_FLAGS "/FS")

    string(REGEX REPLACE "/W[1-3]" "/W4" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REGEX REPLACE "/W[1-3]" "/W4" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})

    if (STK_WARNINGS_ARE_ERRORS)
        set(EXTRA_FLAGS "${extra_flags} /WX")
    endif()

    if (STK_BUILD_WITH_DEBUG_INFO)
        set(EXTRA_FLAGS_RELEASE "/Zi")
        set(EXTRA_LINK_FLAGS_RELEASE "/DEBUG")
    endif()

    if (STK_ENABLE_FAST_MATH)
        set(EXTRA_FLAGS "${extra_flags} /fp:fast")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(EXTRA_FLAGS "-Wall -Wextra -pedantic -Wno-missing-field-initializers")

    if (STK_WARNINGS_ARE_ERRORS)
        set(EXTRA_FLAGS "${extra_flags} -Werror")
    endif()

    if (STK_BUILD_WITH_DEBUG_INFO)
        set(EXTRA_FLAGS_RELEASE "-g1")
    endif()

    if (STK_ENABLE_FAST_MATH)
        set(EXTRA_FLAGS "${extra_flags} -ffast-math")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(EXTRA_FLAGS "-Wall -Wextra -pedantic")

    if (STK_WARNINGS_ARE_ERRORS)
        set(EXTRA_FLAGS "${extra_flags} -Werror")
    endif()

    if (STK_BUILD_WITH_DEBUG_INFO)
        set(EXTRA_FLAGS_RELEASE "-g1")
    endif()

    if (STK_ENABLE_FAST_MATH)
        set(EXTRA_FLAGS "${extra_flags} -ffast-math")
    endif()
endif()

if (STK_USE_CUDA)
    add_definitions(-DSTK_USE_CUDA)
    enable_language(CUDA)

    if (STK_ENABLE_FAST_MATH)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
    endif()
endif()

if (STK_LOGGING_PREFIX_FILE)
    add_definitions(-DSTK_LOGGING_PREFIX_FILE)
endif()



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_FLAGS_RELEASE}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${EXTRA_FLAGS_RELEASE}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_LINK_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${EXTRA_LINK_FLAGS_RELEASE}")

# TODO: Determine QNANHIBIT for NrrdIO
set(QNANHIBIT 1)


if(NOT WIN32)
    find_package(ZLIB QUIET)
    find_package(NIFTI QUIET)
    find_package(NrrdIO QUIET)
endif()

add_subdirectory(${THIRD_PARTY_DIR})
add_subdirectory(src)

if (STK_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if (STK_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

