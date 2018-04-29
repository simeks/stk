add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE Catch2)

add_subdirectory(zlib-1.2.11)
target_include_directories(zlib INTERFACE zlib-1.2.11 ${zlib_BINARY_DIR})

add_subdirectory(nifticlib-2.0.0)
set_target_properties(niftiio PROPERTIES COMPILE_FLAGS 
    "/wd4100 /wd4244 /wd4245 /wd4267 /wd4304 /wd4456"
    )