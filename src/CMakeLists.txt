set(STK_SRCS 
    "stk/common/log.h"
    "stk/common/log.cpp"
)

add_library(stk STATIC ${STK_SRCS})

target_include_directories(stk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

