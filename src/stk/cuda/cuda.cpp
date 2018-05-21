#include "cuda.h"
#include "stk/common/error.h"
#include "stk/common/log.h"

#ifdef STK_USE_CUDA

#include <cuda_runtime.h>

namespace stk
{
    void cuda::init()
    {
        int n = device_count();
        if (n == 0) {
            LOG(Warning) << "[CUDA] No CUDA enabled devices found";
            return;
        }

        for (int i = 0; i < n; ++i) {
            cudaDeviceProp prop;
            CUDA_CHECK_ERROR(cudaGetDeviceProperties(&prop, i));

            LOG(Info) << "[CUDA] Device: " << i << " name: " << prop.name;
        }
        set_device(0);
    }
    int cuda::device_count()
    {
        int n;
        CUDA_CHECK_ERROR(cudaGetDeviceCount(&n));
        return n;
    }
    void cuda::set_device(int device_id)
    {
        CUDA_CHECK_ERROR(cudaSetDevice(device_id));
    }
    int cuda::device()
    {
        int device_id;
        CUDA_CHECK_ERROR(cudaGetDevice(&device_id));
        return device_id;
    }
    void cuda::reset_device()
    {
        CUDA_CHECK_ERROR(cudaDeviceReset());
    }
}

#endif // STK_USE_CUDA
