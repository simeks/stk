#pragma once

#ifdef __CUDACC__
    #define CUDA_HOST_DEVICE __host__ __device__
#else
    #define CUDA_HOST_DEVICE
#endif

#ifdef STK_USE_CUDA

#include <cuda_runtime.h>

#include <stk/common/error.h>

#define CUDA_CHECK_ERRORS(val) \
    if (val != cudaSuccess) { \
        FATAL() << "[CUDA] " << cudaGetErrorString(val) << "(code=" << val << ")"; \
    }

namespace stk
{
    namespace cuda
    {
        // Initializes CUDA
        void init();

        // Returns the number of CUDA-enabled devices
        int device_count();

        // Sets the active cuda device
        void set_device(int device_id);

        // Returns the index to the currently active device
        int device();

        // Resets the current device
        void reset_device();
    }
}
#endif // STK_USE_CUDA
