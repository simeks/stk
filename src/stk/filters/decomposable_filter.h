#pragma once

#include <stk/common/error.h>
#include <stk/image/volume.h>
#include <stk/math/float3.h>

#include <vector>


namespace stk
{
/*!
 * \brief Class representing a filter kernel.
 *
 * \tparam T Type of the filter kernel entries.
 */
template <typename T>
class FilterKernel
{
public:

    /*!
     * \brief Create a filter kernel from a vector.
     * \param entries Vector of filter entries, sorted from
     *                `-radius` to `radius`.
     * \note The number of entries must be odd.
     */
    FilterKernel(const std::vector<T>& entries)
        : _kernel (entries)
        , _radius ((int)entries.size() / 2)
    {
        if (0 == entries.size() % 2) {
            LOG(Error) << "Expected odd number of kernel entries";
        }
    }

    ~FilterKernel() {}

    /*!
     * \brief Radius of the filter kernel. 
     */
    int radius(void) const {
        return _radius; 
    }

    /*!
     * \brief Entry in position `k` w.r.t. the centre.
     * \param k Coordinate of the entry, within the range
     *          `[-radius, radius]`.
     */
    T operator[](const int k) const {
        return _kernel.at(_radius + k); // bound-checked
    }

private:
    std::vector<T> _kernel;
    int _radius;
};


template <typename T>
struct FilterKernel3
{
    FilterKernel<T> x, y, z;
};


/*!
 * \brief Compute a decomposable filtering operation.
 *
 * This function supports different input and output types, allowing for
 * example to have a filter producing an output of different type from
 * the input, or even to compute multiple filters (using a kernel with
 * vectorial components), returning a vector image with one filter
 * response per channel.
 *
 * \tparam TVoxelType  Input volume voxel type.
 * \tparam TKernelType Type of the kernel entries.
 * \tparam TOutputType Output volume voxel type.
 *
 * \param img         Image to be filtered.
 * \param kernel      Three components of the decomposed kernel.
 * \param border_mode Policy on image boundary.
 */
template <typename TKernelType, typename TOutputType>
Volume decomposable_filter_3d(
        const Volume& volume,
        const FilterKernel3<TKernelType> kernel,
        const BorderMode border_mode
        );


/*!
 * \brief Overloaded version with output type matching the input type.
 */
template <typename TKernelType>
Volume decomposable_filter_3d(
        const Volume& volume,
        const FilterKernel3<TKernelType> kernel,
        const BorderMode border_mode
        );


} // namespace stk

#include "decomposable_filter.inl"

