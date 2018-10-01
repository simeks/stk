#include "itk_io.h"
#include "itk_convert.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageIOBase.h>


/*!
 * \brief Read a specific image type with ITK.
 */
template<class ImageType>
stk::Volume read(const itk::ImageIOBase::Pointer image_io)
{
    auto reader = itk::ImageFileReader<ImageType>::New();
    reader->SetImageIO(image_io);
    reader->SetFileName(image_io->GetFileName());

    try {
        reader->Update();
    }
    catch (const itk::ExceptionObject& e) {
        FATAL() << "Error while reading image '" << image_io->GetFileName() << "': "
                << e.what();
    }

    auto image = ImageType::New();
    image->Graft(reader->GetOutput());
    image->SetMetaDataDictionary(reader->GetMetaDataDictionary());

    return stk::itk2stk<ImageType>(image);
}


/*!
 * \brief Read a vectorial image of unspecified type with ITK.
 */
template<unsigned int Dimension, unsigned int Components>
stk::Volume read_vector_image(const itk::ImageIOBase::Pointer image_io)
{
    switch(image_io->GetComponentType())
    {
    case itk::ImageIOBase::UCHAR: {
        using PixelType = itk::Vector<unsigned char, Components>;
        using ImageType = itk::Image<PixelType, Dimension>;
        return read<ImageType>(image_io);
    }
    case itk::ImageIOBase::CHAR: {
        using PixelType = itk::Vector<char, Components>;
        using ImageType = itk::Image<PixelType, Dimension>;
        return read<ImageType>(image_io);
    }
    case itk::ImageIOBase::USHORT: {
        using PixelType = itk::Vector<unsigned short, Components>;
        using ImageType = itk::Image<PixelType, Dimension>;
        return read<ImageType>(image_io);
    }
    case itk::ImageIOBase::SHORT: {
        using PixelType = itk::Vector<short, Components>;
        using ImageType = itk::Image<PixelType, Dimension>;
        return read<ImageType>(image_io);
    }
    case itk::ImageIOBase::UINT: {
        using PixelType = itk::Vector<unsigned int, Components>;
        using ImageType = itk::Image<PixelType, Dimension>;
        return read<ImageType>(image_io);
    }
    case itk::ImageIOBase::INT: {
        using PixelType = itk::Vector<int, Components>;
        using ImageType = itk::Image<PixelType, Dimension>;
        return read<ImageType>(image_io);
    }
    case itk::ImageIOBase::ULONG: {
        using PixelType = itk::Vector<unsigned long, Components>;
        using ImageType = itk::Image<PixelType, Dimension>;
        return read<ImageType>(image_io);
    }
    case itk::ImageIOBase::LONG: {
        using PixelType = itk::Vector<long, Components>;
        using ImageType = itk::Image<PixelType, Dimension>;
        return read<ImageType>(image_io);
    }
    case itk::ImageIOBase::ULONGLONG: {
        using PixelType = itk::Vector<unsigned long long, Components>;
        using ImageType = itk::Image<PixelType, Dimension>;
        return read<ImageType>(image_io);
    }
    case itk::ImageIOBase::LONGLONG: {
        using PixelType = itk::Vector<long long, Components>;
        using ImageType = itk::Image<PixelType, Dimension>;
        return read<ImageType>(image_io);
    }
    case itk::ImageIOBase::FLOAT: {
        using PixelType = itk::Vector<float, Components>;
        using ImageType = itk::Image<PixelType, Dimension>;
        return read<ImageType>(image_io);
    }
    case itk::ImageIOBase::DOUBLE: {
        using PixelType = itk::Vector<double, Components>;
        using ImageType = itk::Image<PixelType, Dimension>;
        return read<ImageType>(image_io);
    }
    case itk::ImageIOBase::UNKNOWNCOMPONENTTYPE:
    default:
        FATAL() << "Cannot read image '" << image_io->GetFileName() << "': "
                << "unknown component type";
    }
    FATAL() << "This line should be unreachable";
}


/*!
 * \brief Read a vectorial image of unspecified type with ITK.
 */
template<unsigned int Dimension>
stk::Volume read_vector_image(const itk::ImageIOBase::Pointer image_io)
{
    switch (image_io->GetNumberOfComponents())
    {
    case 1:
        return read_vector_image<Dimension, 1>(image_io);
    case 2:
        return read_vector_image<Dimension, 2>(image_io);
    case 3:
        return read_vector_image<Dimension, 3>(image_io);
    case 4:
        return read_vector_image<Dimension, 4>(image_io);
    default:
        FATAL() << "Cannot read image '" << image_io->GetFileName() << "': "
                << "unsupported number of components "
                << image_io->GetNumberOfComponents();
    }
    FATAL() << "This line should be unreachable";
}


/*!
 * \brief Read an unknown image with ITK.
 */
stk::Volume stk::read_itk_image(const std::string& file_name)
{
    itk::ImageIOBase::Pointer image_io = itk::ImageIOFactory::CreateImageIO(
            file_name.c_str(),
            itk::ImageIOFactory::ReadMode
            );

    if (!image_io) {
        FATAL() << "Cannot read image '" << file_name <<"': "
                << "the format is unrecognised by ITK";
    }

    image_io->SetFileName(file_name);
    image_io->ReadImageInformation();

    switch(image_io->GetPixelType())
    {
    case itk::ImageIOBase::SCALAR:
    case itk::ImageIOBase::VECTOR: {
        switch (image_io->GetNumberOfDimensions())
        {
        case 3:
            return read_vector_image<3>(image_io);
        default:
            FATAL() << "Cannot read image '" << file_name << "': "
                    << "image dimension '" << image_io->GetNumberOfDimensions() << "' "
                    << "not supported";
        }
    }

    case itk::ImageIOBase::COMPLEX:
    case itk::ImageIOBase::COVARIANTVECTOR:
    case itk::ImageIOBase::DIFFUSIONTENSOR3D:
    case itk::ImageIOBase::FIXEDARRAY:
    case itk::ImageIOBase::MATRIX:
    case itk::ImageIOBase::OFFSET:
    case itk::ImageIOBase::POINT:
    case itk::ImageIOBase::RGB:
    case itk::ImageIOBase::RGBA:
    case itk::ImageIOBase::SYMMETRICSECONDRANKTENSOR:
        FATAL() << "Cannot read image '" << file_name << "': "
                << "unsupported pixel type '" << image_io->GetPixelType() << "'";

    case itk::ImageIOBase::UNKNOWNPIXELTYPE:
        FATAL() << "Cannot read image '" << file_name << "': "
                << "unknown pixel type";
    }

    FATAL() << "This line should be unreachable";
}


/*!
 * \brief Write a specific image type with ITK.
 */
template<class ImageType>
void write(const stk::Volume& volume, const std::string& file_name)
{
    typename ImageType::Pointer image;
    stk::stk2itk<ImageType>(volume, image);

    auto writer = itk::ImageFileWriter<ImageType>::New();
    writer->SetFileName(file_name);
    writer->SetInput(image);

    try {
        writer->Update();
    }
    catch (const itk::ExceptionObject& e) {
        FATAL() << "Error while writing image '" << file_name << "': "
                << e.what();
    }
}


/*!
 * \brief Write a specific image type with ITK.
 */
template<int Components>
void write(const stk::Volume& volume, const std::string& file_name)
{
    switch (stk::base_type(volume.voxel_type()))
    {
    case stk::Type_UChar: {
        using PixelType = itk::Vector<unsigned char, Components>;
        using ImageType = itk::Image<PixelType, 3>;
        write<ImageType>(volume, file_name);
        break;
    }
    case stk::Type_Char: {
        using PixelType = itk::Vector<char, Components>;
        using ImageType = itk::Image<PixelType, 3>;
        write<ImageType>(volume, file_name);
        break;
    }
    case stk::Type_UShort: {
        using PixelType = itk::Vector<unsigned short, Components>;
        using ImageType = itk::Image<PixelType, 3>;
        write<ImageType>(volume, file_name);
        break;
    }
    case stk::Type_Short: {
        using PixelType = itk::Vector<short, Components>;
        using ImageType = itk::Image<PixelType, 3>;
        write<ImageType>(volume, file_name);
        break;
    }
    case stk::Type_UInt: {
        using PixelType = itk::Vector<unsigned int, Components>;
        using ImageType = itk::Image<PixelType, 3>;
        write<ImageType>(volume, file_name);
        break;
    }
    case stk::Type_Int: {
        using PixelType = itk::Vector<int, Components>;
        using ImageType = itk::Image<PixelType, 3>;
        write<ImageType>(volume, file_name);
        break;
    }
    case stk::Type_Float: {
        using PixelType = itk::Vector<float, Components>;
        using ImageType = itk::Image<PixelType, 3>;
        write<ImageType>(volume, file_name);
        break;
    }
    case stk::Type_Double: {
        using PixelType = itk::Vector<double, Components>;
        using ImageType = itk::Image<PixelType, 3>;
        write<ImageType>(volume, file_name);
        break;
    }
    default:
        FATAL() << "This line should be unreachable";
    }
}


/*!
 * \brief Write an image to file with ITK.
 */
void stk::write_itk_image(const Volume& volume, const std::string& file_name)
{
    switch (stk::num_components(volume.voxel_type()))
    {
    case 1:
        write<1>(volume, file_name);
        break;
    case 2:
        write<2>(volume, file_name);
        break;
    case 3:
        write<3>(volume, file_name);
        break;
    case 4:
        write<4>(volume, file_name);
        break;
    default:
        FATAL() << "Unsupported number of components";
    }
}
