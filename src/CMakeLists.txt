set(STK_SRCS 
    "stk/common/error.cpp"
    "stk/common/error.h"
    "stk/common/log.cpp"
    "stk/common/log.h"
    "stk/math/types.h"
    "stk/volume/dim3.h"
)

add_library(stk STATIC ${STK_SRCS})

target_include_directories(stk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

