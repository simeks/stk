# stk
Toolkit for 3D volume processing

# Build

```bash
mkdir build && cd build
cmake ..
make
```

CMake options:
  + `STK_BUILD_EXAMPLES`: build example programs
  + `STK_BUILD_TESTS`: build tests
  + `STK_USE_CUDA`: build with CUDA support
  + `STK_WARNINGS_ARE_ERRORS`: compilation will fail on warnings
  + `STK_BUILD_WITH_DEBUG_INFO`: include debug symbols in the binaries
  + `STK_ENABLE_FAST_MATH`: enable unsafe (non IEEE 754-compliant) optimisations
  + `STK_LOGGING_PREFIX_FILE`: add the file name as prefix to each log message

When building with `STK_USE_CUDA`, in case the version of `gcc` selected by
CMake was not compatible with the one required by CUDA, it is possible to
specify a different executable with `-DCMAKE_CUDA_FLAGS="-ccbin gcc-XX"`, where
`gcc-XX` is a version of `gcc` compatible with your CUDA version.

