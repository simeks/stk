if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    string(REPLACE "-Wextra" ""  CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "-Wextra" ""  CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "-Wextra" ""  CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

    string(REPLACE "-pedantic" ""  CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "-pedantic" ""  CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "-pedantic" ""  CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
endif()

add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/Catch2>
  $<INSTALL_INTERFACE:include>
)

if(NOT ZLIB_FOUND)
    add_subdirectory(zlib)
    target_include_directories(zlib INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/zlib>
        $<BUILD_INTERFACE:${zlib_BINARY_DIR}>
        $<INSTALL_INTERFACE:include>
    )
    target_include_directories(zlibstatic INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/zlib>
        $<BUILD_INTERFACE:${zlib_BINARY_DIR}>
        $<INSTALL_INTERFACE:include>
    )

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

    if(NOT ZLIB_FOUND)
        # NIFTI requires ZLIB, use bundled version if not found.
        set(NIFTI_ZLIB_LIBRARIES ZLIB::ZLIBSTATIC CACHE STRING "" FORCE)
    endif()

    add_subdirectory(nifti_clib)
    set(NIFTILIB_INCLUDE_DIRS "nifti_clib/niftilib")

    target_include_directories(niftiio INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/nifti_clib/niftilib>
        $<INSTALL_INTERFACE:include>
    )
    target_include_directories(znz INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/nifti_clib/znzlib>
        $<INSTALL_INTERFACE:include>
    )

    if (ZLIB_FOUND)
        target_include_directories(znz PUBLIC "${ZLIB_INCLUDE_DIRS}")
    endif()

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
    add_subdirectory(NrrdIO)
 
    target_include_directories(NrrdIO INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/NrrdIO>
        $<INSTALL_INTERFACE:include>
    )

    if (ZLIB_FOUND)
        target_include_directories(NrrdIO PRIVATE "${ZLIB_INCLUDE_DIRS}")
    endif()

    if (WIN32)
        set_target_properties(NrrdIO PROPERTIES COMPILE_FLAGS
            "/wd4127 /wd4701 /wd4706 /wd4996"
            )
    endif()
endif()

if(STK_BUILD_PYTHON_WRAPPER)
    add_subdirectory(pybind11)
endif()

