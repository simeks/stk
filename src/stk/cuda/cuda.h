#pragma once

#define CUDA_CHECK_ERRORS(val) \
    if (val != cudaSuccess) { \
        cudaDeviceReset(); \
        FATAL() << "CUDA error: " << cudaGetErrorString(val); \
    }

namespace stk
{
#ifdef STK_USE_CUDA
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
#endif // STK_USE_CUDA
}