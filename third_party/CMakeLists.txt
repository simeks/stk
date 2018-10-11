if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    string(REPLACE "-Wextra" ""  CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "-Wextra" ""  CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "-Wextra" ""  CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

    string(REPLACE "-pedantic" ""  CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "-pedantic" ""  CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "-pedantic" ""  CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
endif()

add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE Catch2)

if(NOT ZLIB_FOUND)
    add_subdirectory(zlib-1.2.11)
    target_include_directories(zlib INTERFACE zlib-1.2.11 ${zlib_BINARY_DIR})
    target_include_directories(zlibstatic INTERFACE zlib-1.2.11 ${zlib_BINARY_DIR})

    if (WIN32)
        set_target_properties(zlib PROPERTIES COMPILE_FLAGS
            "/wd4100 /wd4127 /wd4131 /wd4244 /wd4245 /wd4267 /wd4305"
            )
        set_target_properties(zlibstatic PROPERTIES COMPILE_FLAGS
            "/wd4100 /wd4127 /wd4131 /wd4244 /wd4245 /wd4267 /wd4305"
            )
        set_target_properties(zlibstatic PROPERTIES COMPILE_FLAGS
            "/wd4100 /wd4127 /wd4131 /wd4244 /wd4245 /wd4267 /wd4305"
            )
    endif()
endif()


if(NOT NIFTI_FOUND)
    add_subdirectory(nifticlib-2.0.0)
    set(NIFTI_INCLUDE_DIRS "nifticlib-2.0.0/niftilib")
    target_include_directories(niftiio INTERFACE "nifticlib-2.0.0/niftilib")
    target_include_directories(znz INTERFACE "nifticlib-2.0.0/znzlib")
    if (WIN32)
        set_target_properties(niftiio PROPERTIES COMPILE_FLAGS
            "/wd4100 /wd4244 /wd4245 /wd4267 /wd4304 /wd4456 /wd4701"
            )
        set_target_properties(znz PROPERTIES COMPILE_FLAGS
            "/wd4100 /wd4244 /wd4245 /wd4267 /wd4304 /wd4456"
            )
    endif()

    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set_target_properties(niftiio PROPERTIES COMPILE_FLAGS
            "-Wno-misleading-indentation -Wno-unused-but-set-variable"
            )
    endif()
endif()

if(NOT NrrdIO_FOUND)
    add_subdirectory(NrrdIO-1.11.0)
    target_include_directories(NrrdIO INTERFACE "NrrdIO-1.11.0")
    if (WIN32)
        set_target_properties(NrrdIO PROPERTIES COMPILE_FLAGS
            "/wd4127 /wd4701 /wd4706 /wd4996"
            )
    endif()
endif()

