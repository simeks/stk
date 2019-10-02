pybind11_add_module(_stk _stk.cpp)

target_include_directories(_stk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYTHON_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIR} ${pybind11_INCLUDE_DIR})
target_link_libraries(_stk PUBLIC stk)

set(STK_PYTHON_PATH "$ENV{PYTHONPATH}${SEP}${CMAKE_CURRENT_BINARY_DIR}" CACHE STRING "Python path for Sphinx")

