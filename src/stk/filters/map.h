#pragma once

#include <stk/image/volume.h>

#include <functional>

namespace stk {

template<typename TVoxelInputType, typename TVoxelOutputType>
VolumeHelper<TVoxelOutputType> map(
        const stk::VolumeHelper<TVoxelInputType>& image,
        const std::function<TVoxelOutputType(TVoxelInputType)> op,
        VolumeHelper<TVoxelOutputType> *out = nullptr
        )
{
    VolumeHelper<TVoxelOutputType> dest;
    if (!out) {
        dest.allocate(image.size());
        out = &dest;
    }
    ASSERT(out->size() == image.size());
    out->copy_meta_from(image);

    #pragma omp parallel for
    for (int z = 0; z < (int) image.size().z; ++z) {
        for (int y = 0; y < (int) image.size().y; ++y) {
            for (int x = 0; x < (int) image.size().x; ++x) {
                (*out)(x, y, z) = op(image(x, y, z));
            }
        }
    }

    return *out;
}

} // namespace stk

