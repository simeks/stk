#include "sobel.h"

stk::Volume stk::sobel(const stk::Volume& volume)
{
    const float3 s = {
        1 / volume.spacing().x,
        1 / volume.spacing().y,
        1 / volume.spacing().z
    };

    FilterKernel3<float3> kernels = {
        FilterKernel<float3>({{-s.x, 1.0f, 1.0f}, {0.0f, 2.0f, 2.0f}, { s.x, 1.0f, 1.0f}}),
        FilterKernel<float3>({{1.0f, -s.y, 1.0f}, {2.0f, 0.0f, 2.0f}, {1.0f,  s.y, 1.0f}}),
        FilterKernel<float3>({{1.0f, 1.0f, -s.z}, {2.0f, 2.0f, 0.0f}, {1.0f, 1.0f,  s.z}}),
    };

    return decomposable_filter_3d<float3, float3>(volume,
                                                  kernels,
                                                  stk::Border_Replicate);
}

