set(STK_SRCS 
    "stk/common/error.cpp"
    "stk/common/error.h"
    "stk/common/log.cpp"
    "stk/common/log.h"
    "stk/cuda/volume.h"
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

if (STK_USE_CUDA)
    cuda_add_library(stk STATIC ${STK_SRCS})
else()
    add_library(stk STATIC ${STK_SRCS})
endif()

if(NIFTI_FOUND)
    target_include_directories(stk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_INCLUDE_DIRS} ${NIFTI_INCLUDE_DIR})
    target_link_libraries(stk NrrdIO ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${NIFTI_LIBRARIES})
else()
    target_include_directories(stk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_INCLUDE_DIRS})
    target_link_libraries(stk niftiio NrrdIO  ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
endif()
