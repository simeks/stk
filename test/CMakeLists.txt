set(STK_TEST_SRCS 
    "test_dim3.cpp"
    "test_error.cpp"
    "test_filters.cpp"
    "test_io.cpp"
    "test_main.cpp"
    "test_math.cpp"
    "test_logging.cpp"
    "test_types.cpp"
    "test_util.h"
    "test_volume.cpp"
    )

if (STK_USE_CUDA)
    set(STK_TEST_SRCS ${STK_TEST_SRCS} "test_cuda.cpp")
endif ()


add_executable(stk_test ${STK_TEST_SRCS})
add_test(stk_test stk_test)

target_link_libraries(stk_test Catch stk)

