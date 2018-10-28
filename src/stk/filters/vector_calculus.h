#pragma once

#include <stk/image/volume.h>

namespace stk {

/*!
 * \brief Compute the gradient of a given image.
 * @param image Input image.
 * @param out Optional object for the output. If not `nullptr`,
 *            the filter is executed in place, otherwise a new
 *            volume is allocated. The object must have the same
 *            size as the input, and a compatible vector type.
 */
template<typename TVoxelType, typename TOutputType = float3>
VolumeHelper<TOutputType> nabla(
        const VolumeHelper<TVoxelType>& image,
        VolumeHelper<TOutputType> *out = nullptr
        );

/*!
 * \brief Compute the divergence of a vector field.
 * @param vf  Input 3D 3-vector field.
 * @param out Optional object for the output. If not `nullptr`,
 *            the filter is executed in place, otherwise a new
 *            volume is allocated. The object must have the same
 *            size as the input.
 */
template<typename TVoxelType, typename TOutputType = float>
VolumeHelper<TOutputType> divergence(
        const VolumeHelper<TVoxelType>& vf,
        VolumeHelper<TOutputType> *out = nullptr
        );

/*!
 * \brief Compute the gradient of a vector field.
 * @param vf  Input 3D 3-vector field.
 * @param out Optional object for the output. If not `nullptr`,
 *            the filter is executed in place, otherwise a new
 *            volume is allocated. The object must have the same
 *            size as the input, and a compatible vector type.
 */
template<typename TVoxelType, typename TOutputType = float3>
VolumeHelper<TOutputType> rotor(
        const VolumeHelper<TVoxelType>& vf,
        VolumeHelper<TOutputType> *out = nullptr
        );

/*!
 * \brief Compute the gradient of a vector field.
 * @param vf  Input 3D 3-vector field.
 * @param out Optional object for the output. If not `nullptr`,
 *            the filter is executed in place, otherwise a new
 *            volume is allocated. The object must have the same
 *            size as the input.
 */
template<typename TVoxelType, typename TOutputType = float>
VolumeHelper<TOutputType> circulation_density(
        const VolumeHelper<TVoxelType>& vf,
        VolumeHelper<TOutputType> *out = nullptr
        );

} // namespace stk

#include "vector_calculus.inl"

