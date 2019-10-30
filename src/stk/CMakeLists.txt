set(STK_SRCS
    "common/error.cpp"
    "common/error.h"
    "common/log.cpp"
    "common/log.h"
    "common/stack_trace.cpp"
    "common/stack_trace.h"
    "cuda/volume.h"
    "filters/decomposable_filter.h"
    "filters/decomposable_filter.inl"
    "filters/gaussian_filter.cpp"
    "filters/gaussian_filter.h"
    "filters/map.h"
    "filters/sobel.cpp"
    "filters/sobel.h"
    "filters/vector_calculus.h"
    "filters/vector_calculus.inl"
    "io/io.cpp"
    "io/io.h"
    "io/nifti.cpp"
    "io/nifti.h"
    "io/nrrd.cpp"
    "io/nrrd.h"
    "io/vtk.cpp"
    "io/vtk.h"
    "math/float2.h"
    "math/float3.h"
    "math/float4.h"
    "math/math.cpp"
    "math/math.h"
    "math/types.h"
    "image/dim3.h"
    "image/types.cpp"
    "image/types.h"
    "image/volume.cpp"
    "image/volume.h"
    "image/volume.inl"
)

if (STK_USE_CUDA)
    set(STK_SRCS
        ${STK_SRCS}
        "cuda/cuda.cpp"
        "cuda/cuda.h"
        "cuda/stream.cpp"
        "cuda/stream.h"
        "cuda/volume.cpp"
        "cuda/volume.h"
        "filters/gpu/normalize.cu"
        "image/gpu_volume.cpp"
        "image/gpu_volume.cu"
        "image/gpu_volume.h"
    )
endif()

if(STK_ITK_BRIDGE)
    set(STK_SRCS
        ${STK_SRCS}
        "itk_bridge/itk_convert.h"
        "itk_bridge/itk_io.cpp"
        "itk_bridge/itk_io.h"
        )
endif()

add_library(stk STATIC ${STK_SRCS})

target_include_directories(stk PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

if(NIFTI_FOUND)
    target_include_directories(stk PUBLIC ${NIFTI_INCLUDE_DIR})
    target_link_libraries(stk PUBLIC ${NIFTI_LIBRARIES})
else()
    target_link_libraries(stk PUBLIC niftiio)
endif()

if(NrrdIO_FOUND)
    target_include_directories(stk PUBLIC ${NrrdIO_INCLUDE_DIR})
    target_link_libraries(stk PUBLIC ${NrrdIO_LIBRARIES})
else()
    target_link_libraries(stk PUBLIC NrrdIO)
endif()

if(STK_ITK_BRIDGE)
    add_definitions(-DSTK_ITK_BRIDGE)
    find_package(ITK REQUIRED)
    include(${ITK_USE_FILE})

    target_include_directories(stk PRIVATE ${ITK_INCLUDE_DIRS})
    target_link_libraries(stk PRIVATE ${ITK_LIBRARIES})
endif()

if(STK_STACK_TRACE)
    add_definitions(-DSTK_STACK_TRACE)

    if(UNIX)
        find_package(Backward)
        if(Backward_FOUND)
            add_backward(stk)
            target_link_libraries(stk PUBLIC ${BACKWARD_LIBRARIES})
            add_definitions(-DSTK_USE_BACKWARD)
        endif()
    endif()
endif()

