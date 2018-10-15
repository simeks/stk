#ifdef STK_USE_CUDA

#include "gpu_volume.h"
#include "volume.h"

#include "stk/cuda/cuda.h"
#include "stk/cuda/stream.h"

#include <cuda_runtime.h>

using namespace stk;

namespace
{
    bool create_format_desc(Type voxel_type, cudaChannelFormatDesc& desc)
    {
        switch (voxel_type) {
        case Type_Char:
            desc = cudaCreateChannelDesc<char1>();
            return true;
        case Type_Char2:
            desc = cudaCreateChannelDesc<char2>();
            return true;
        case Type_Char4:
            desc = cudaCreateChannelDesc<char4>();
            return true;

        case Type_UChar:
            desc = cudaCreateChannelDesc<uchar1>();
            return true;
        case Type_UChar2:
            desc = cudaCreateChannelDesc<uchar2>();
            return true;
        case Type_UChar4:
            desc = cudaCreateChannelDesc<uchar4>();
            return true;

        case Type_Short:
            desc = cudaCreateChannelDesc<short1>();
            return true;
        case Type_Short2:
            desc = cudaCreateChannelDesc<short2>();
            return true;
        case Type_Short4:
            desc = cudaCreateChannelDesc<short4>();
            return true;

        case Type_UShort:
            desc = cudaCreateChannelDesc<ushort1>();
            return true;
        case Type_UShort2:
            desc = cudaCreateChannelDesc<ushort2>();
            return true;
        case Type_UShort4:
            desc = cudaCreateChannelDesc<ushort4>();
            return true;

        case Type_Int:
            desc = cudaCreateChannelDesc<int1>();
            return true;
        case Type_Int2:
            desc = cudaCreateChannelDesc<int2>();
            return true;
        case Type_Int4:
            desc = cudaCreateChannelDesc<int4>();
            return true;

        case Type_UInt:
            desc = cudaCreateChannelDesc<uint1>();
            return true;
        case Type_UInt2:
            desc = cudaCreateChannelDesc<uint2>();
            return true;
        case Type_UInt4:
            desc = cudaCreateChannelDesc<uint4>();
            return true;

        case Type_Float:
            desc = cudaCreateChannelDesc<float>();
            return true;
        case Type_Float2:
            desc = cudaCreateChannelDesc<float2>();
            return true;
        case Type_Float4:
            desc = cudaCreateChannelDesc<float4>();
            return true;

        default:
            // Note: float3, etc are not supported, use float4 or 3 separate float volumes
            FATAL() << "Unsupported channel format (voxel_type=" << voxel_type << ")";
            return false;
        };
    }

    std::shared_ptr<GpuVolumeData>
    allocate_gpu_volume(const dim3& size, Type voxel_type, gpu::Usage usage)
    {
        auto vol = std::make_shared<GpuVolumeData>();
        vol->usage  = usage;

        if (!create_format_desc(voxel_type, vol->format_desc))
            return nullptr;

        // Extent width is defined in terms of elements if any cudaArray is present,
        //  otherwise in number of bytes (for pitched pointer)

        if (vol->usage == gpu::Usage_PitchedPointer) {
            // Size in bytes
            size_t per_voxel = (vol->format_desc.x + vol->format_desc.y + vol->format_desc.z + vol->format_desc.w) / 8;
            cudaExtent extent = make_cudaExtent(size.x * per_voxel, size.y, size.z);
            CUDA_CHECK_ERRORS(cudaMalloc3D(&vol->pitched_ptr, extent));
        }
        else {
            cudaExtent extent = make_cudaExtent(size.x, size.y, size.z);
            // TODO: What about cudaArraySurfaceLoadStore?
            CUDA_CHECK_ERRORS(cudaMalloc3DArray(&vol->array_ptr, &vol->format_desc, extent));
        }

        return vol;
    }

    cudaMemcpy3DParms make_download_params(const GpuVolume& src, const Volume& dst)
    {
        cudaMemcpy3DParms params = {};

        // We also assume both volumes have same dimensions
        ASSERT(dst.size() == src.size());
        ASSERT(dst.voxel_type() == src.voxel_type());

        dim3 size = dst.size();

        params.kind = cudaMemcpyDeviceToHost;
        // xsize in bytes
        params.dstPtr = make_cudaPitchedPtr(const_cast<void*>(dst.ptr()),
            dst.strides()[1], dst.strides()[1], dst.strides()[2]/dst.strides()[1]);

        // Extent width is defined in terms of elements if any cudaArray is present,
        //  otherwise in number of bytes (for pitched pointer)
        size_t per_voxel = 1;
        if (src.usage() == gpu::Usage_PitchedPointer) {
            params.srcPtr = src.pitched_ptr();
            per_voxel = type_size(src.voxel_type());
        }
        else {
            params.srcArray = src.array_ptr();
            per_voxel = 1;
        }

        params.extent = make_cudaExtent(size.x * per_voxel, size.y, size.z);
        return params;
    }

    cudaMemcpy3DParms make_upload_params(const Volume& src, const GpuVolume& dst)
    {
        cudaMemcpy3DParms params = {};

        // We also assume both volumes have same dimensions
        ASSERT(src.size() == dst.size());
        ASSERT(src.voxel_type() == dst.voxel_type());

        dim3 size = src.size();

        params.kind = cudaMemcpyHostToDevice;
        // xsize in bytes
        params.srcPtr = make_cudaPitchedPtr(const_cast<void*>(src.ptr()),
            src.strides()[1], src.strides()[1], src.strides()[2]/src.strides()[1]);

        // Extent width is defined in terms of elements if any cudaArray is present,
        //  otherwise in number of bytes (for pitched pointer)
        size_t per_voxel = 1;
        if (dst.usage() == gpu::Usage_PitchedPointer) {
            params.dstPtr = dst.pitched_ptr();
            per_voxel = type_size(dst.voxel_type());
        }
        else {
            params.dstArray = dst.array_ptr();
            per_voxel = 1;
        }
        params.extent = make_cudaExtent(size.x * per_voxel, size.y, size.z);
        return params;
    }

    cudaMemcpy3DParms make_d2d_params(const GpuVolume& src, const GpuVolume& dst)
    {
        cudaMemcpy3DParms params = {};

        ASSERT(src.size() == dst.size());
        ASSERT(src.voxel_type() == src.voxel_type());

        params.kind = cudaMemcpyDeviceToDevice;

        // Extent width is defined in terms of elements if any cudaArray is present,
        //  otherwise in number of bytes (for pitched pointer)
        size_t per_voxel = 1;
        if (src.usage() == gpu::Usage_PitchedPointer && dst.usage() == gpu::Usage_PitchedPointer) {
            per_voxel = type_size(src.voxel_type());
        }

        if (src.usage() == gpu::Usage_PitchedPointer) {
            params.srcPtr = src.pitched_ptr();
        }
        else {
            params.srcArray = src.array_ptr();
        }

        if (dst.usage() == gpu::Usage_PitchedPointer) {
            params.dstPtr = dst.pitched_ptr();
        }
        else {
            params.dstArray = dst.array_ptr();
        }
        params.extent = make_cudaExtent(dst.size().x * per_voxel, dst.size().y, dst.size().z);
        return params;
    }
}

namespace stk
{
namespace gpu
{
    Type voxel_type(const cudaChannelFormatDesc& format_desc)
    {
        int num_comp = 0;
        if (format_desc.x > 0) ++num_comp;
        if (format_desc.y > 0) ++num_comp;
        if (format_desc.z > 0) ++num_comp;
        if (format_desc.w > 0) ++num_comp;

        Type base_type = Type_Unknown;
        if (format_desc.f == cudaChannelFormatKindFloat) {
            if (format_desc.x != 32) {
                FATAL() << "Unsupported format";
            }

            base_type = Type_Float;
        }
        else if (format_desc.f == cudaChannelFormatKindSigned) {
            if (format_desc.x == 8) {
                base_type = Type_Char;
            }
            else if (format_desc.x == 16) {
                base_type = Type_Short;
            }
            if (format_desc.x == 32) {
                base_type = Type_Int;
            }
        }
        else if (format_desc.f == cudaChannelFormatKindUnsigned) {
            if (format_desc.x == 8) {
                base_type = Type_UChar;
            }
            else if (format_desc.x == 16) {
                base_type = Type_UShort;
            }
            if (format_desc.x == 32) {
                base_type = Type_UInt;
            }
        }

        Type type = build_type(base_type, num_comp);
        FATAL_IF(type == Type_Unknown)
            << "Unsupported format";

        return type;
    }
}

GpuVolumeData::GpuVolumeData() :
    usage(gpu::Usage_PitchedPointer),
    array_ptr(nullptr)
{
    format_desc = {};
    pitched_ptr = {};
}
GpuVolumeData::~GpuVolumeData()
{
    if (usage == gpu::Usage_PitchedPointer) {
        if (pitched_ptr.ptr == nullptr)
            return; // not allocated;

                    // No error checks for free as that will cause problems if an error
                    //  has already been triggered and the system is shutting down.
        cudaFree(pitched_ptr.ptr);
        pitched_ptr.ptr = nullptr;
    }
    else {
        if (array_ptr == nullptr)
            return; // not allocated;

        cudaFreeArray(array_ptr);
        array_ptr = nullptr;
    }
}

GpuVolume::GpuVolume() :
    _size{0,0,0},
    _origin{0,0,0},
    _spacing{1,1,1},
    _ptr{}
{
    _direction.diagonal({1, 1, 1});
    _inverse_direction.diagonal({1, 1, 1});
}
GpuVolume::GpuVolume(const GpuVolume& other) :
    _data(other._data),
    _size(other._size),
    _origin(other._origin),
    _spacing(other._spacing),
    _direction(other._direction),
    _inverse_direction(other._inverse_direction),
    _ptr(other._ptr)
{
}
GpuVolume& GpuVolume::operator=(const GpuVolume& other)
{
    if (this != &other) {
        _data = other._data;
        _size = other._size;
        _origin = other._origin;
        _spacing = other._spacing;
        _direction = other._direction;
        _inverse_direction = other._inverse_direction;
        _ptr = other._ptr;
    }
    return *this;
}
GpuVolume::GpuVolume(const dim3& size, Type voxel_type, gpu::Usage usage) :
    _origin({0,0,0}),
    _spacing({1,1,1}),
    _ptr({})
{
    _direction.diagonal({1, 1, 1});
    _inverse_direction.diagonal({1, 1, 1});
    allocate(size, voxel_type, usage);
}
GpuVolume::GpuVolume(const GpuVolume& other, const Range& x, const Range& y, const Range& z) :
    _data(other._data),
    _spacing(other._spacing),
    _direction(other._direction),
    _inverse_direction(other._inverse_direction),
    _ptr(other._ptr)
{
    // Subvolumes are only supported for pitched pointer types
    ASSERT(other.usage() == gpu::Usage_PitchedPointer);
    ASSERT(other.valid());
    ASSERT(x.begin <= x.end && y.begin <= y.end && z.begin <= z.end);
    ASSERT(0 <= x.begin && x.end <= (int)other._size.x);
    ASSERT(0 <= y.begin && y.end <= (int)other._size.y);
    ASSERT(0 <= z.begin && z.end <= (int)other._size.z);

    int nx = x.end - x.begin;
    int ny = y.end - y.begin;
    int nz = z.end - z.begin;

    uint8_t* ptr = reinterpret_cast<uint8_t*>(_ptr.ptr);

    if (z.begin != 0 && z.end != (int)_size.z) {
        // any offset in z axis does not break contiguity
        ptr += z.begin * _ptr.pitch * _ptr.ysize;
    }

    if (y.begin != 0 && y.end != (int)_size.y) {
        ptr += y.begin * _ptr.pitch;
    }

    if (x.begin != 0 && x.end != (int)_size.x) {
        ptr += x.begin * type_size(voxel_type());
    }

    _size = dim3{(uint32_t)nx, (uint32_t)ny, (uint32_t)nz};
    _ptr = make_cudaPitchedPtr(ptr, _ptr.pitch, _ptr.xsize, _ptr.ysize);
    
    _origin = {
        other._origin.x + _spacing.x * x.begin,
        other._origin.y + _spacing.y * y.begin,
        other._origin.z + _spacing.z * z.begin
    };
}
GpuVolume::~GpuVolume()
{
}
void GpuVolume::release()
{
    _data = nullptr;
}
GpuVolume GpuVolume::clone() const
{
    GpuVolume copy(_size, voxel_type(), usage());
    copy.copy_from(*this);
    return copy;
}
GpuVolume GpuVolume::clone(const cuda::Stream& stream) const
{
    GpuVolume copy(_size, voxel_type(), usage());
    copy.copy_from(*this, stream);
    return copy;
}
GpuVolume GpuVolume::clone_as(gpu::Usage usage) const
{
    GpuVolume copy(_size, voxel_type(), usage);
    copy.copy_from(*this);
    return copy;
}
GpuVolume GpuVolume::clone_as(gpu::Usage usage, const cuda::Stream& stream) const
{
    GpuVolume copy(_size, voxel_type(), usage);
    copy.copy_from(*this, stream);
    return copy;
}
void GpuVolume::copy_from(const GpuVolume& other)
{
    ASSERT(valid());
    ASSERT(other.valid());

    cudaMemcpy3DParms params = make_d2d_params(other, *this);
    CUDA_CHECK_ERRORS(cudaMemcpy3D(&params));

    copy_meta_from(other);
}
void GpuVolume::copy_from(const GpuVolume& other, const cuda::Stream& stream)
{
    ASSERT(valid());
    ASSERT(other.valid());

    cudaMemcpy3DParms params = make_d2d_params(other, *this);
    CUDA_CHECK_ERRORS(cudaMemcpy3DAsync(&params, stream));

    copy_meta_from(other);
}
bool GpuVolume::valid() const
{
    return _data &&
        (_data->pitched_ptr.ptr != nullptr || _data->array_ptr != nullptr);
}
dim3 GpuVolume::size() const
{
    return _size;
}
void GpuVolume::set_origin(const float3& origin)
{
    _origin = origin;
}
void GpuVolume::set_spacing(const float3& spacing)
{
    _spacing = spacing;
}
void GpuVolume::set_direction(const Matrix3x3f& direction)
{
    _direction = direction;
    _inverse_direction = _direction.inverse();
}
void GpuVolume::set_direction(const std::initializer_list<float> direction)
{
    _direction.set(direction);
    _inverse_direction = _direction.inverse();
}

const float3& GpuVolume::origin() const
{
    return _origin;
}
const float3& GpuVolume::spacing() const
{
    return _spacing;
}
const Matrix3x3f& GpuVolume::direction() const
{
    return _direction;
}
const Matrix3x3f& GpuVolume::inverse_direction() const
{
    return _inverse_direction;
}
void GpuVolume::copy_meta_from(const GpuVolume& other)
{
    _origin = other._origin;
    _spacing = other._spacing;
    _direction = other._direction;
    _inverse_direction = other._inverse_direction;
}

Type GpuVolume::voxel_type() const
{
    return _data ? gpu::voxel_type(_data->format_desc) : Type_Unknown;
}

GpuVolume::GpuVolume(const Volume& vol, gpu::Usage usage)
{
    allocate(vol.size(), vol.voxel_type(), usage);
    upload(vol);

    set_origin(vol.origin());
    set_spacing(vol.spacing());
    set_direction(vol.direction());
}
GpuVolume::GpuVolume(const Volume& vol, const cuda::Stream& stream, gpu::Usage usage)
{
    allocate(vol.size(), vol.voxel_type(), usage);
    upload(vol, stream);

    set_origin(vol.origin());
    set_spacing(vol.spacing());
    set_direction(vol.direction());
}
Volume GpuVolume::download() const
{
    // Requires gpu memory to be allocated
    if (!valid())
        return Volume();

    Volume vol(_size, voxel_type());
    download(vol);
    return vol;
}
Volume GpuVolume::download(const cuda::Stream& stream) const
{
    // Requires gpu memory to be allocated
    if (!valid())
        return Volume();

    Volume vol(_size, voxel_type());
    download(vol, stream);
    return vol;
}
void GpuVolume::download(Volume& vol) const
{
    ASSERT(valid()); // Requires gpu memory to be allocated
    ASSERT(vol.valid()); // Requires cpu memory to be allocated as well

    cudaMemcpy3DParms params = make_download_params(*this, vol);
    CUDA_CHECK_ERRORS(cudaMemcpy3D(&params));

    vol.set_origin(_origin);
    vol.set_spacing(_spacing);
    vol.set_direction(_direction);
}
void GpuVolume::download(Volume& vol, const cuda::Stream& stream) const
{
    ASSERT(valid()); // Requires gpu memory to be allocated
    ASSERT(vol.valid()); // Requires cpu memory to be allocated as well

    cudaMemcpy3DParms params = make_download_params(*this, vol);
    CUDA_CHECK_ERRORS(cudaMemcpy3DAsync(&params, stream));

    vol.set_origin(_origin);
    vol.set_spacing(_spacing);
    vol.set_direction(_direction);
}
void GpuVolume::upload(const Volume& vol)
{
    ASSERT(valid()); // Requires gpu memory to be allocated
    ASSERT(vol.valid()); // Requires cpu memory to be allocated as well

    cudaMemcpy3DParms params = make_upload_params(vol, *this);
    CUDA_CHECK_ERRORS(cudaMemcpy3D(&params));

    set_origin(vol.origin());
    set_spacing(vol.spacing());
    set_direction(vol.direction());
}
void GpuVolume::upload(const Volume& vol, const cuda::Stream& stream)
{
    ASSERT(valid()); // Requires gpu memory to be allocated
    ASSERT(vol.valid()); // Requires cpu memory to be allocated as well

    // We also assume both volumes have same dimensions
    ASSERT(vol.size() == _size);
    ASSERT(vol.voxel_type() == voxel_type());

    cudaMemcpy3DParms params = make_upload_params(vol, *this);
    CUDA_CHECK_ERRORS(cudaMemcpy3DAsync(&params, stream));

    set_origin(vol.origin());
    set_spacing(vol.spacing());
    set_direction(vol.direction());
}
gpu::Usage GpuVolume::usage() const
{
    ASSERT(valid());
    return _data->usage;
}
GpuVolume GpuVolume::as_usage(gpu::Usage usage) const
{
    ASSERT(valid());
    if (_data->usage == usage)
        return *this;

    GpuVolume copy(_size, voxel_type(), usage);
    copy.copy_from(*this); // Copies meta as well

    return copy;
}
cudaArray* GpuVolume::array_ptr() const
{
    ASSERT(valid());
    ASSERT(_data);
    ASSERT(usage() == gpu::Usage_Texture);
    return _data->array_ptr;
}
cudaPitchedPtr GpuVolume::pitched_ptr() const
{
    ASSERT(valid());
    ASSERT(_data);
    ASSERT(usage() == gpu::Usage_PitchedPointer);
    return _ptr;
}
GpuVolume GpuVolume::operator()(const Range& x, const Range& y, const Range& z)
{
    return GpuVolume(*this, x, y, z);
}
void GpuVolume::allocate(const dim3& size, Type voxel_type, gpu::Usage usage)
{
    _size = size;
    _data = allocate_gpu_volume(size, voxel_type, usage);
    if (usage == gpu::Usage_PitchedPointer)
        _ptr = _data->pitched_ptr;
}
} // namespace stk

#endif // STK_USE_CUDA
