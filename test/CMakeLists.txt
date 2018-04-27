set(STK_TEST_SRCS 
    "test_dim3.cpp"
    "test_error.cpp"
    "test_main.cpp"
    "test_math.cpp"
    "test_logging.cpp"
    "test_types.cpp"
    "test_volume.cpp"
    )

add_executable(stk_test ${STK_TEST_SRCS})
add_test(stk_test stk_test)

target_link_libraries(stk_test Catch stk)

