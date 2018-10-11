#pragma once

namespace
{

/*!
 * \brief Auxiliary function.
 */
template<typename TVoxelType, typename TKernelType, typename TOutputType = TVoxelType>
stk::VolumeHelper<TOutputType> decomposable_filter_3d(
        const stk::VolumeHelper<TVoxelType>& img,
        const stk::FilterKernel3<TKernelType>& kernel,
        const stk::BorderMode border_mode
        )
{
    stk::VolumeHelper<TOutputType> result(img.size(), TOutputType{});
    result.copy_meta_from(img);

    const int3 dims = {int(img.size().x), int(img.size().y), int(img.size().z)};

    // X dimension
    #pragma omp parallel for
    for (int y = 0; y < dims.y; ++y)
    {
        for (int z = 0; z < dims.z; ++z)
        {
            for (int x = 0; x < dims.x; ++x)
            {
                TOutputType value{};
                for (int t = -kernel.x.radius(); t <= kernel.x.radius(); t++)
                {
                    const TVoxelType v = img.at(x + t, y, z, border_mode);
                    value += TOutputType(v * kernel.x[t]);
                }
                result(x, y, z) = value;
            }
        }
    }

    stk::VolumeHelper<TOutputType> tmp = result.clone();

    //Y dimension
    #pragma omp parallel for
    for (int x = 0; x < dims.x; ++x)
    {
        for (int z = 0; z < dims.z; ++z)
        {
            for (int y = 0; y < dims.y; ++y)
            {
                TOutputType value{};
                for (int t = -kernel.y.radius(); t <= kernel.y.radius(); t++)
                {
                    const TOutputType v = tmp.at(x, y + t, z, border_mode);
                    value += TOutputType(v * kernel.y[t]);
                }
                result(x, y, z) = value;
            }
        }
    }

    tmp.copy_from(result);

    //Z dimension
    #pragma omp parallel for
    for (int x = 0; x < dims.x; ++x)
    {
        for (int y = 0; y < dims.y; ++y)
        {
            for (int z = 0; z < dims.z; ++z)
            {
                TOutputType value{};
                for (int t = -kernel.z.radius(); t <= kernel.z.radius(); t++)
                {
                    const TOutputType v = tmp.at(x, y, z + t, border_mode);
                    value += TOutputType(v * kernel.z[t]);
                }
                result(x, y, z) = value;
            }
        }
    }
    return result;
}

} // namespace


namespace stk {

template <typename TKernelType, typename TOutputType>
Volume decomposable_filter_3d(
            const Volume& volume,
            const FilterKernel3<TKernelType> kernel,
            const BorderMode border_mode
            )
{
    switch (volume.voxel_type())
    {
    case Type_Float:
        return ::decomposable_filter_3d<float, TKernelType, TOutputType>(
                volume, {kernel.x, kernel.y, kernel.z}, border_mode);
    case Type_Double:
        return ::decomposable_filter_3d<double, TKernelType, TOutputType>(
                volume, {kernel.x, kernel.y, kernel.z}, border_mode);
    case Type_Float3:
        return ::decomposable_filter_3d<float3, TKernelType, TOutputType>(
                volume, {kernel.x, kernel.y, kernel.z}, border_mode);
    default:
        FATAL() << "Unsupported voxel format";
    };
    return Volume();
}


template <typename TKernelType>
Volume decomposable_filter_3d(
            const Volume& volume,
            const FilterKernel3<TKernelType> kernel,
            const BorderMode border_mode
            )
{
    switch (volume.voxel_type())
    {
    case Type_Float:
        return ::decomposable_filter_3d<float, TKernelType, float>(
                volume, {kernel.x, kernel.y, kernel.z}, border_mode);
    case Type_Double:
        return ::decomposable_filter_3d<double, TKernelType, double>(
                volume, {kernel.x, kernel.y, kernel.z}, border_mode);
    case Type_Float3:
        return ::decomposable_filter_3d<float3, TKernelType, float3>(
                volume, {kernel.x, kernel.y, kernel.z}, border_mode);
    default:
        FATAL() << "Unsupported voxel format";
    };
    return Volume();
}


} // namespace stk

