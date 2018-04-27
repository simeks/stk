set(STK_SRCS 
    "stk/common/error.cpp"
    "stk/common/error.h"
    "stk/common/log.cpp"
    "stk/common/log.h"
    "stk/io/io.cpp"
    "stk/io/io.h"
    "stk/io/vtk.cpp"
    "stk/io/vtk.h"
    "stk/math/types.h"
    "stk/volume/dim3.h"
    "stk/volume/types.cpp"
    "stk/volume/types.h"
    "stk/volume/volume.cpp"
    "stk/volume/volume.h"
    "stk/volume/volume.inl"
)

add_library(stk STATIC ${STK_SRCS})

target_include_directories(stk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

