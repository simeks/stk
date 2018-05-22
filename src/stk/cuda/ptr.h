#pragma once

namespace stk
{
    namespace cuda
    {
        template<typename T>
        struct VolumePtr
        {
            VolumePtr(const cudaPitchedPtr& ptr) : 
                ptr((T*)ptr.ptr),
                pitch(ptr.pitch),
                ysize(ptr.ysize)
            {
            }

            __device__ T& operator()(int x, int y, int z)
            { 
                return ((T*)(((uint8_t*)ptr) + (y * pitch + z * pitch * ysize)))[x];
            }
            __device__ const T& operator()(int x, int y, int z) const
            {
                return ((T*)(((uint8_t*)ptr) + (y * pitch + z * pitch * ysize)))[x];
            }

            T* ptr;
            size_t pitch;
            size_t ysize;
        };
    }
}