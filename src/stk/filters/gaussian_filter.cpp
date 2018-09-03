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

    auto kernel = stk::gaussian_kernel(sigma);
    stk::FilterKernel3<float> kernels = {kernel, kernel, kernel};

    return stk::decomposable_filter_3d<float>(img, kernels, stk::Border_Mirror);
}

} // namespace


stk::FilterKernel<float> stk::gaussian_kernel(const float sigma)
{
    int r = (int) std::ceil(2 * sigma); // filter radius
    std::vector<float> kernel (2 * r + 1);

    const float k = -1.0 / (2.0 * sigma * sigma);
    float sum = 0.0;
    for (int i = 0; i < 2*r + 1; ++i) {
        kernel[i] = std::exp(k * (i - r) * (i - r));
        sum += kernel[i];
    }

    // Normalise
    for (int i = 0; i < 2*r + 1; ++i) {
        kernel[i] /= sum;
    }

    return FilterKernel<float>(kernel);
}

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

