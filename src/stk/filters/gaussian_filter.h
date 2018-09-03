#pragma once

#include <stk/image/volume.h>

namespace stk
{
    /*!
     * \brief Generate a 1D Gaussian kernel.
     * \param sigma Standard deviation.
     * \return A Gaussian filter kernel with radius `r` and size `2r + 1`.
     */
    FilterKernel<float> gaussian_kernel(const float sigma);

    /// Gaussian filter for 3d volumes
    /// Performs per-component filtering for multi-channel volumes.
    Volume gaussian_filter_3d(const Volume& volume, float sigma);
}
