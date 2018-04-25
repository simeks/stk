set(STK_TEST_SRCS 
    "test_error.cpp"
    "test_main.cpp"
    "test_math.cpp"
    "test_logging.cpp"
    )

add_executable(stk_test ${STK_TEST_SRCS})
add_test(stk_test stk_test)

target_link_libraries(stk_test Catch stk)

