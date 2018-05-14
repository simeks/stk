set(STK_SRCS 
    "stk/common/error.cpp"
    "stk/common/error.h"
    "stk/common/log.cpp"
    "stk/common/log.h"
    "stk/io/io.cpp"
    "stk/io/io.h"
    "stk/io/nifti.cpp"
    "stk/io/nifti.h"
    "stk/io/nrrd.cpp"
    "stk/io/nrrd.h"
    "stk/io/vtk.cpp"
    "stk/io/vtk.h"
    "stk/math/types.h"
    "stk/image/dim3.h"
    "stk/image/types.cpp"
    "stk/image/types.h"
    "stk/image/volume.cpp"
    "stk/image/volume.h"
    "stk/image/volume.inl"
)

add_library(stk STATIC ${STK_SRCS})

target_include_directories(stk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(stk niftiio NrrdIO)

