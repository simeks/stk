#pragma once

#include <stk/common/assert.h>
#include <stk/image/gpu_volume.h>
#include <stk/image/types.h>

namespace stk
{
    namespace cuda
    {
        template<typename T>
        struct VolumePtr
        {
            VolumePtr(const GpuVolume& vol) : 
                ptr((T*)vol.pitched_ptr().ptr),
                pitch(vol.pitched_ptr().pitch),
                ysize(vol.pitched_ptr().ysize)
            {
                ASSERT(vol.voxel_type() == type_id<T>::id);
                ASSERT(vol.usage() == gpu::Usage_PitchedPointer);
            }

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