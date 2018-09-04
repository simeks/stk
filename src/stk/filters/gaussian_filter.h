#pragma once

#include <stk/image/volume.h>
#include <stk/filters/decomposable_filter.h>

#include <cmath>
#include <vector>

namespace stk
{

/*!
 * \brief Generate a 1D Gaussian kernel.
 * \param sigma Standard deviation.
 * \return A Gaussian filter kernel with radius `r` and size `2r + 1`.
 */
template <typename T>
FilterKernel<T> gaussian_kernel(const T sigma, const float spacing)
{
    int r = (int) std::ceil(3 * sigma / spacing); // filter radius
    std::vector<T> kernel (2 * r + 1);

    const T k = -1.0f / (2.0f * sigma * sigma);
    T sum = 0.0;
    for (int i = 0; i < 2*r + 1; ++i) {
        kernel[i] = std::exp(k * (i - r) * (i - r));
        sum += kernel[i];
    }

    // Normalise
    for (int i = 0; i < 2*r + 1; ++i) {
        kernel[i] /= sum;
    }

    return FilterKernel<T>(kernel);
}


/// Gaussian filter for 3d volumes
/// Performs per-component filtering for multi-channel volumes.
Volume gaussian_filter_3d(const Volume& volume, float sigma);

}
