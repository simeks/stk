#pragma once

#include "map.h"

namespace stk {

template<typename TVoxelType, typename TOutputType>
VolumeHelper<TOutputType> nabla(
        const VolumeHelper<TVoxelType>& image,
        VolumeHelper<TOutputType> *out
        )
{
    VolumeHelper<TOutputType> dest;
    if (!out) {
        dest.allocate(image.size());
        out = &dest;
    }
    ASSERT(out->size() == image.size());
    ASSERT(num_components(out->voxel_type()) == 3);
    out->copy_meta_from(image);

    const float3 s = float3{0.5, 0.5, 0.5} / image.spacing();
    const BorderMode bm = Border_Replicate;
    using T = typename type_id<TOutputType>::Base;

    #pragma omp parallel for
    for (int z = 0; z < (int) image.size().z; ++z) {
        for (int y = 0; y < (int) image.size().y; ++y) {
            for (int x = 0; x < (int) image.size().x; ++x) {
                (*out)(x, y, z) = TOutputType{
                    s.x * T(image.at(x+1, y,   z,   bm) - image.at(x-1, y,   z,   bm)),
                    s.y * T(image.at(x,   y+1, z,   bm) - image.at(x,   y-1, z,   bm)),
                    s.z * T(image.at(x,   y,   z+1, bm) - image.at(x,   y,   z-1, bm))
                };
            }
        }
    }

    return *out;
}

template<typename TVoxelType, typename TOutputType>
VolumeHelper<TOutputType> divergence(
        const VolumeHelper<TVoxelType>& vf,
        VolumeHelper<TOutputType> *out
        )
{
    VolumeHelper<TOutputType> dest;
    if (!out) {
        dest.allocate(vf.size());
        out = &dest;
    }
    ASSERT(out->size() == vf.size());
    ASSERT(num_components(out->voxel_type()) == 1);
    out->copy_meta_from(vf);

    const float3 s = float3{0.5, 0.5, 0.5} / vf.spacing();
    const BorderMode bm = Border_Replicate;
    using T = TOutputType;

    #pragma omp parallel for
    for (int z = 0; z < (int) vf.size().z; ++z) {
        for (int y = 0; y < (int) vf.size().y; ++y) {
            for (int x = 0; x < (int) vf.size().x; ++x) {
                (*out)(x, y, z) =
                    s.x * (T(vf.at(x+1, y,   z,   bm).x) - T(vf.at(x-1, y,   z,   bm).x)) +
                    s.y * (T(vf.at(x,   y+1, z,   bm).y) - T(vf.at(x,   y-1, z,   bm).y)) +
                    s.z * (T(vf.at(x,   y,   z+1, bm).z) - T(vf.at(x,   y,   z-1, bm).z));
            }
        }
    }

    return *out;
}

template<typename TVoxelType, typename TOutputType>
VolumeHelper<TOutputType> rotor(
        const VolumeHelper<TVoxelType>& vf,
        VolumeHelper<TOutputType> *out
        )
{
    VolumeHelper<TOutputType> dest;
    if (!out) {
        dest.allocate(vf.size());
        out = &dest;
    }
    ASSERT(out->size() == vf.size());
    ASSERT(num_components(out->voxel_type()) == 3);
    out->copy_meta_from(vf);

    const float3 s = float3{0.5, 0.5, 0.5} / vf.spacing();
    const BorderMode bm = Border_Replicate;
    using T = typename type_id<TOutputType>::Base;

    #pragma omp parallel for
    for (int z = 0; z < (int) vf.size().z; ++z) {
        for (int y = 0; y < (int) vf.size().y; ++y) {
            for (int x = 0; x < (int) vf.size().x; ++x) {
                T dx_dy = s.y * T(vf.at(x,   y+1, z,   bm).x - vf.at(x,   y-1, z,   bm).x);
                T dx_dz = s.z * T(vf.at(x,   y,   z+1, bm).x - vf.at(x,   y,   z-1, bm).x);
                T dy_dx = s.x * T(vf.at(x+1, y,   z,   bm).y - vf.at(x-1, y,   z,   bm).y);
                T dy_dz = s.z * T(vf.at(x,   y,   z+1, bm).y - vf.at(x,   y,   z-1, bm).y);
                T dz_dx = s.x * T(vf.at(x+1, y,   z,   bm).z - vf.at(x-1, y,   z,   bm).z);
                T dz_dy = s.y * T(vf.at(x,   y+1, z,   bm).z - vf.at(x,   y-1, z,   bm).z);
                (*out)(x, y, z) = TOutputType{dz_dy - dy_dz, dx_dz - dz_dx, dy_dx - dx_dy};
            }
        }
    }

    return *out;
}

template<typename TVoxelType, typename TOutputType>
VolumeHelper<TOutputType> circulation_density(
        const VolumeHelper<TVoxelType>& vf,
        VolumeHelper<TOutputType> *out
        )
{
    return map<float3, float>(rotor(vf), norm, out);
}

} // namespace stk
