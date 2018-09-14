#pragma once

#include "decomposable_filter.h"

namespace stk {

/*!
 * \brief Compute the Sobel operator along dimension `dim`.
 * \return A scalar volume containing the smooth derivative along axis `dim`.
 */
template <int dim>
stk::Volume sobel(const stk::Volume& volume)
{
    static_assert(dim >= 0 && dim < 3, "Invalid dimension");

    const std::vector<float> s = {
        1 / volume.spacing().x,
        1 / volume.spacing().y,
        1 / volume.spacing().z
    };

    std::vector<FilterKernel<float>> kernels;
    for (int i = 0; i < 3; ++i) {
        kernels.push_back(dim == i ? FilterKernel<float>({-s[i], 0.0f, s[i]})
                                   : FilterKernel<float>({ 1.0f, 2.0f, 1.0f}));
    }

    return decomposable_filter_3d<float>(volume,
                                         {kernels[0], kernels[1], kernels[2]},
                                         stk::Border_Replicate);
}


/*!
 * \brief Compute the Sobel operator along all dimensions.
 * \return A vector volume containing the smooth gradient of the image.
 */
stk::Volume sobel(const stk::Volume& volume);

} // namespace stk
