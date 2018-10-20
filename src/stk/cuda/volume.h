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
                ptr(vol.valid() ? (T*)vol.pitched_ptr().ptr : nullptr),
                pitch(vol.valid() ? vol.pitched_ptr().pitch : 0),
                ysize(vol.valid() ? vol.pitched_ptr().ysize : 0)
            {
                if (vol.valid()) {
                    ASSERT(vol.voxel_type() == type_id<T>::id());
                    ASSERT(vol.usage() == gpu::Usage_PitchedPointer);
                }
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
            int x1 = int(x);
            int y1 = int(y);
            int z1 = int(z);

            int x2 = min(x1+1, int(dims.x-1));
            int y2 = min(y1+1, int(dims.y-1));
            int z2 = min(z1+1, int(dims.z-1));

            if (x1 < 0 || x1 >= int(dims.x) ||
                y1 < 0 || y1 >= int(dims.y) ||
                z1 < 0 || z1 >= int(dims.z))
            {
                return T{0};
            }

            float xt = x - x1;
            float yt = y - y1;
            float zt = z - z1;

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
        template<typename T>
        __device__ T linear_at_clamp(const VolumePtr<T>& vol, const dim3& dims,
                                     float x, float y, float z)
        {
            // Clamp
            x = max(0.0f, min(x, (float)dims.x-1));
            y = max(0.0f, min(y, (float)dims.y-1));
            z = max(0.0f, min(z, (float)dims.z-1));

            int x1 = int(x);
            int y1 = int(y);
            int z1 = int(z);

            int x2 = min(x1+1, int(dims.x-1));
            int y2 = min(y1+1, int(dims.y-1));
            int z2 = min(z1+1, int(dims.z-1));

            float xt = x - x1;
            float yt = y - y1;
            float zt = z - z1;

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
