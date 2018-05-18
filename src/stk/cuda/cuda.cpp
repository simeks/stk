#include "cuda.h"

#ifdef STK_USE_CUDA

#include <cuda_runtime.h>


namespace stk
{
    void cuda::init(int device_id)
    {
        CUDA_CHECK_ERROR(cudaSetDevice(device_id));
        CUDA_CHECK_ERROR(cudaGetDevice(&device_id));

        cudaDeviceProp device_prop;
        CUDA_CHECK_ERROR(cudaGetDeviceProperties(&device_prop, device_id));

        LOG(Info) << "[CUDA] Device " << device_id << ", name: " << device_prop.name;
    }
}

#endif // STK_USE_CUDA
