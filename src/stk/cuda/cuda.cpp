#include "cuda.h"
#include "stk/common/assert.h"
#include "stk/common/error.h"
#include "stk/common/log.h"

#ifdef STK_USE_CUDA

#include <cstring>
#include <cuda_runtime.h>

namespace stk {
namespace cuda {
    void init()
    {
        int n = device_count();
        if (n == 0) {
            LOG(Warning) << "[CUDA] No CUDA enabled devices found";
            return;
        }

        for (int i = 0; i < n; ++i) {
            cudaDeviceProp prop;
            CUDA_CHECK_ERRORS(cudaGetDeviceProperties(&prop, i));

            LOG(Info) << "[CUDA] Device: " << i << " name: " << prop.name;
        }
        set_device(0);
    }
    int device_count()
    {
        int n;
        CUDA_CHECK_ERRORS(cudaGetDeviceCount(&n));
        return n;
    }
    void set_device(int device_id)
    {
        CUDA_CHECK_ERRORS(cudaSetDevice(device_id));
    }
    int device()
    {
        int device_id;
        CUDA_CHECK_ERRORS(cudaGetDevice(&device_id));
        return device_id;
    }
    void reset_device()
    {
        CUDA_CHECK_ERRORS(cudaDeviceReset());
    }

} // namespace cuda
} // namespace stk

#endif // STK_USE_CUDA
