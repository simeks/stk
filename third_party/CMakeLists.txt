add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE Catch2)

add_subdirectory(zlib-1.2.11)
target_include_directories(zlib INTERFACE zlib-1.2.11 ${zlib_BINARY_DIR})
target_include_directories(zlibstatic INTERFACE zlib-1.2.11 ${zlib_BINARY_DIR})

set_target_properties(zlib PROPERTIES COMPILE_FLAGS 
    "/wd4100 /wd4127 /wd4131 /wd4244 /wd4245 /wd4267 /wd4305"
    )
set_target_properties(zlibstatic PROPERTIES COMPILE_FLAGS 
    "/wd4100 /wd4127 /wd4131 /wd4244 /wd4245 /wd4267 /wd4305"
    )
set_target_properties(zlibstatic PROPERTIES COMPILE_FLAGS 
    "/wd4100 /wd4127 /wd4131 /wd4244 /wd4245 /wd4267 /wd4305"
    )

add_subdirectory(nifticlib-2.0.0)
target_include_directories(niftiio INTERFACE "nifticlib-2.0.0/niftilib")
target_include_directories(znz INTERFACE "nifticlib-2.0.0/znzlib")
set_target_properties(niftiio PROPERTIES COMPILE_FLAGS 
    "/wd4100 /wd4244 /wd4245 /wd4267 /wd4304 /wd4456 /wd4701"
    )
set_target_properties(znz PROPERTIES COMPILE_FLAGS 
    "/wd4100 /wd4244 /wd4245 /wd4267 /wd4304 /wd4456"
    )

add_subdirectory(NrrdIO-1.11.0)
target_include_directories(NrrdIO INTERFACE "NrrdIO-1.11.0")
set_target_properties(NrrdIO PROPERTIES COMPILE_FLAGS
    "/wd4127 /wd4701 /wd4706 /wd4996"
    )