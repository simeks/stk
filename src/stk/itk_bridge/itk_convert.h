#pragma once

#include <stk/image/volume.h>

#include <itkImageIOBase.h>

namespace stk {


/*!
 * \brief Convert an ITK type to the corresponding STK type.
 */
template<typename TPixelType>
stk::Type itk2stk_type(void)
{
    return stk::type_id<typename itk::PixelTraits<TPixelType>::ValueType>::id();
}


/*!
 * \brief Convert an ITK object into an STK object.
 * \note The new object owns a deep copy of the image data.
 */
template<class ImageType>
stk::Volume itk2stk(const typename ImageType::Pointer & image)
{
    auto const region = image->GetLargestPossibleRegion();
    auto const size = region.GetSize();

    const dim3 stk_size = {uint32_t(size[0]), uint32_t(size[1]), uint32_t(size[2])};
    const size_t voxel_no = size[0] * size[1] * size[2];
    const stk::Type stk_type = itk2stk_type<typename ImageType::PixelType>();

    if (stk::Type_Unknown == stk_type) {
        FATAL() << "Conversion error, unknown or unsupported image type";
    }

    stk::Volume volume (stk_size, stk_type);

    // Copy image data
    std::memcpy(volume.ptr(),
                image->GetBufferPointer(),
                sizeof (typename ImageType::PixelType) * voxel_no);

    // Copy metadata
    const auto spacing = image->GetSpacing();
    volume.set_spacing({float(spacing[0]), float(spacing[1]), float(spacing[2])});
    const auto origin = image->GetOrigin();
    volume.set_origin({float(origin[0]), float(origin[1]), float(origin[2])});

    // Store other metadata
    auto const& metadata = image->GetMetaDataDictionary();
    for (auto const& k : metadata.GetKeys()) {
        volume.set_metadata(k, metadata[k]);
    }

    return volume;
}


/*!
 * \brief Convert an ITK object into an STK object.
 * \note The new object owns a deep copy of the image data.
 */
template<class ImageType>
void stk2itk(const stk::Volume& volume, typename ImageType::Pointer& image)
{
    using IndexType = typename ImageType::IndexType;
    using SizeType = typename ImageType::SizeType;
    using SizeValueType = typename ImageType::SizeValueType;
    using RegionType = typename ImageType::RegionType;
    using PointType = typename ImageType::PointType;
    using PointValueType = typename ImageType::PointValueType;
    using SpacingType = typename ImageType::SpacingType;
    using SpacingValueType = typename ImageType::SpacingValueType;

    const IndexType start = {0, 0, 0};
    const SizeType size = {
        SizeValueType(volume.size().x),
        SizeValueType(volume.size().y),
        SizeValueType(volume.size().z),
    };
    PointType origin;
    origin[0] = PointValueType(volume.origin().x);
    origin[1] = PointValueType(volume.origin().y);
    origin[2] = PointValueType(volume.origin().z);
    SpacingType spacing;
    spacing[0] = SpacingValueType(volume.spacing().x);
    spacing[1] = SpacingValueType(volume.spacing().y);
    spacing[2] = SpacingValueType(volume.spacing().z);

    RegionType region;
    region.SetIndex(start);
    region.SetSize(size);

    image = ImageType::New();
    image->SetRegions(region);
    image->SetOrigin(origin);
    image->SetSpacing(spacing);
    image->Allocate();

    const size_t voxel_no = size[0] * size[1] * size[2];

    // Copy image data
    std::memcpy(image->GetBufferPointer(),
                volume.ptr(),
                sizeof (typename ImageType::PixelType) * voxel_no);

    // Store meta-data
    auto& metadata = image->GetMetaDataDictionary();
    for (auto const& k : volume.get_metadata_keys()) {
        metadata.Set(k, std::any_cast<itk::MetaDataObjectBase::Pointer>(metadata[k]));
    }
}


} // namespace stk
