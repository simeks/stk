#pragma once

#include <stk/math/types.h>
#include <stk/image/volume.h>

#include <itkImageIOBase.h>
#include <itkMetaDataObject.h>

namespace stk {


/*!
 * \brief Convert an ITK type to the corresponding STK type.
 */
template<typename TPixelType>
stk::Type itk2stk_type(void)
{
    return stk::build_type(stk::type_id<typename itk::PixelTraits<TPixelType>::ValueType>::id(),
                           itk::PixelTraits<TPixelType>::Dimension);
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

    const auto direction = image->GetDirection();
    Matrix3x3f stk_direction;
    for (unsigned int i = 0; i < direction.RowDimensions; ++i) {
        for (unsigned int j = 0; j < direction.ColumnDimensions; ++j) {
            stk_direction(i, j) = direction(i, j);
        }
    }
    volume.set_direction(stk_direction);

    // Store other metadata
    auto const& metadata = image->GetMetaDataDictionary();
    for (auto const& k : metadata.GetKeys()) {
        std::string v;
        itk::ExposeMetaData<std::string>(metadata, k, v);
        volume.set_metadata(k, v);
    }

    return volume;
}


/*!
 * \brief Convert an STK object into an ITK object.
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
    using DirectionType = typename ImageType::DirectionType;
    using PixelType = typename ImageType::PixelType;

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
    DirectionType direction;
    auto const stk_direction = volume.direction();
    for (unsigned int i = 0; i < stk_direction.rows; ++i) {
        for (unsigned int j = 0; j < stk_direction.cols; ++j) {
            direction(i, j) = stk_direction(i, j);
        }
    }

    RegionType region;
    region.SetIndex(start);
    region.SetSize(size);

    image = ImageType::New();
    image->SetRegions(region);
    image->SetOrigin(origin);
    image->SetSpacing(spacing);
    image->SetDirection(direction);
    image->Allocate();

    const size_t voxel_no = size[0] * size[1] * size[2];

    // Copy image data
    if (volume.is_contiguous()) {
        std::memcpy(image->GetBufferPointer(),
                    volume.ptr(),
                    sizeof (PixelType) * voxel_no);
    }
    else {
        size_t row_bytes = sizeof (PixelType) * size[0];

        for (int z = 0; z < int(size[2]); ++z) {
            PixelType* dst_row = image->GetBufferPointer() + z * size[0] * size[1];
            const uint8_t* src_row = reinterpret_cast<const uint8_t*>(volume.ptr()) +
                                     z * volume.strides()[2];

            for (int y = 0; y < int(size[1]); ++y) {
                std::memcpy(dst_row, src_row, row_bytes);
                dst_row += size[0];
                src_row += volume.strides()[1];
            }
        }
    }

    // Store meta-data
    auto& metadata = image->GetMetaDataDictionary();
    for (auto const& k : volume.get_metadata_keys()) {
        itk::EncapsulateMetaData<std::string>(metadata, k, volume.get_metadata(k));
    }
    image->SetMetaDataDictionary(metadata);
}


} // namespace stk
