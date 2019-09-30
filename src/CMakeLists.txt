add_subdirectory(stk)

if(STK_BUILD_PYTHON_WRAPPER)
    add_subdirectory(python_wrapper)
endif()
