#pragma once

#define CUDA_CHECK_ERROR(val) \
    if (val) {\
        FATAL() << "CUDA error: " << val;\
    }


namespace stk
{
#ifdef STK_USE_CUDA
    namespace cuda
    {
        void init(int device_id);
    }
#endif // STK_USE_CUDA
}