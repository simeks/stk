set(STK_SRCS
    "stk/common/error.cpp"
    "stk/common/error.h"
    "stk/common/log.cpp"
    "stk/common/log.h"
    "stk/common/stack_trace.cpp"
    "stk/common/stack_trace.h"
    "stk/cuda/volume.h"
    "stk/filters/decomposable_filter.h"
    "stk/filters/decomposable_filter.inl"
    "stk/filters/gaussian_filter.cpp"
    "stk/filters/gaussian_filter.h"
    "stk/filters/map.h"
    "stk/filters/sobel.cpp"
    "stk/filters/sobel.h"
    "stk/filters/vector_calculus.h"
    "stk/filters/vector_calculus.inl"
    "stk/io/io.cpp"
    "stk/io/io.h"
    "stk/io/nifti.cpp"
    "stk/io/nifti.h"
    "stk/io/nrrd.cpp"
    "stk/io/nrrd.h"
    "stk/io/vtk.cpp"
    "stk/io/vtk.h"
    "stk/math/float2.h"
    "stk/math/float3.h"
    "stk/math/float4.h"
    "stk/math/math.cpp"
    "stk/math/math.h"
    "stk/math/types.h"
    "stk/image/dim3.h"
    "stk/image/types.cpp"
    "stk/image/types.h"
    "stk/image/volume.cpp"
    "stk/image/volume.h"
    "stk/image/volume.inl"
)

if (STK_USE_CUDA)
    set(STK_SRCS
        ${STK_SRCS}
        "stk/cuda/cuda.cpp"
        "stk/cuda/cuda.h"
        "stk/cuda/stream.cpp"
        "stk/cuda/stream.h"
        "stk/filters/gpu/normalize.cu"
        "stk/image/gpu_volume.cpp"
        "stk/image/gpu_volume.cu"
        "stk/image/gpu_volume.h"
    )
endif()

if(STK_ITK_BRIDGE)
    set(STK_SRCS
        ${STK_SRCS}
        "stk/itk_bridge/itk_convert.h"
        "stk/itk_bridge/itk_io.cpp"
        "stk/itk_bridge/itk_io.h"
        )
endif()

add_library(stk STATIC ${STK_SRCS})

target_include_directories(stk PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
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

