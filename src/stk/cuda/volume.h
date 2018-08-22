#pragma once

#include <stk/common/assert.h>
#include <stk/image/gpu_volume.h>
#include <stk/image/types.h>

#include <stk/math/float2.h>
#include <stk/math/float3.h>
#include <stk/math/float4.h>

namespace stk
{
    namespace cuda
    {
        // Wrapper around pitched pointers in CUDA
        // This class should be as lean as possible as we typically pass several
        //  VolumePtr which may share properties when invoking a kernel. So no
        //  volume size, origin, and spacing.
        // TODO: Could probably change pitch and ysize to 32bit
        template<typename T>
        struct VolumePtr
        {
            VolumePtr(const GpuVolume& vol) : 
                ptr((T*)vol.pitched_ptr().ptr),
                pitch(vol.pitched_ptr().pitch),
                ysize(vol.pitched_ptr().ysize)
            {
                ASSERT(vol.voxel_type() == type_id<T>::id());
                ASSERT(vol.usage() == gpu::Usage_PitchedPointer);
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

#ifdef __CUDACC__
        template<typename T>
        __device__ T linear_at_border(const VolumePtr<T>& vol, const dim3& dims, 
                                      float x, float y, float z)
        {
            int x1 = int(floorf(x));
            int y1 = int(floorf(y));
            int z1 = int(floorf(z));

            int x2 = int(ceilf(x));
            int y2 = int(ceilf(y));
            int z2 = int(ceilf(z));

            if (x1 < 0 || x2 >= int(dims.x) ||
                y1 < 0 || y2 >= int(dims.y) ||
                z1 < 0 || z2 >= int(dims.z))
            {
                return T{0};
            }

            float xt = x - floorf(x);
            float yt = y - floorf(y);
            float zt = z - floorf(z);

            T s111 = vol(x1, y1, z1);
            T s211 = vol(x2, y1, z1);

            T s121 = vol(x1, y2, z1);
            T s221 = vol(x2, y2, z1);

            T s112 = vol(x1, y1, z2);
            T s212 = vol(x2, y1, z2);

            T s122 = vol(x1, y2, z2);
            T s222 = vol(x2, y2, z2);

            return T(
                (1 - zt) *
                (
                    (1 - yt) *
                    (
                        (1 - xt) * s111 +
                        (xt) * s211
                    ) +

                    (yt) *
                    (
                        (1 - xt) * s121 +
                        (xt) * s221
                    )
                ) +
            (zt) *
                (
                    (1 - yt)*
                    (
                        (1 - xt)*s112 +
                        (xt)*s212
                    ) +

                    (yt)*
                    (
                        (1 - xt)*s122 +
                        (xt)*s222
                    )
                )
            );
        }
#endif // __CUDACC__
    }
}