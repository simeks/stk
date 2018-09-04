#include "decomposable_filter.h"
#include "gaussian_filter.h"

namespace {

template<typename TVoxelType>
stk::VolumeHelper<TVoxelType> gaussian_filter_3d(
        const stk::VolumeHelper<TVoxelType>& img,
        float sigma
        )
{
    ASSERT(sigma >= 0.0f);
    if (sigma <= 0.0f) {
        return img.clone();
    }

    stk::FilterKernel3<float> kernels = {
        stk::gaussian_kernel<float>(sigma, img.spacing().x),
        stk::gaussian_kernel<float>(sigma, img.spacing().y),
        stk::gaussian_kernel<float>(sigma, img.spacing().z),
    };

    return stk::decomposable_filter_3d<float>(img, kernels, stk::Border_Mirror);
}

} // namespace


stk::Volume stk::gaussian_filter_3d(const stk::Volume& volume, float sigma)
{
    switch (volume.voxel_type())
    {
    case stk::Type_Float:
        return ::gaussian_filter_3d<float>(volume, sigma);
    case stk::Type_Double:
        return ::gaussian_filter_3d<double>(volume, sigma);
    case stk::Type_Float3:
        return ::gaussian_filter_3d<float3>(volume, sigma);
    default:
        FATAL() << "Unsupported voxel format";
    };
    return stk::Volume();
}

