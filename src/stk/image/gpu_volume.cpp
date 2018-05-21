#ifdef STK_USE_CUDA

#include "gpu_volume.h"
#include "volume.h"

#include "stk/cuda/cuda.h"

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
            CUDA_CHECK_ERRORS(cudaMalloc3DArray(&vol->array_ptr, &vol->format_desc, extent));
        }

        return vol;
    }
    void release_gpu_volume(GpuVolumeData& vol)
    {
        if (vol.usage == gpu::Usage_PitchedPointer) {
            if (vol.pitched_ptr.ptr == nullptr)
                return; // not allocated;

            CUDA_CHECK_ERRORS(cudaFree(vol.pitched_ptr.ptr));
            vol.pitched_ptr.ptr = nullptr;
        }
        else {
            if (vol.array_ptr == nullptr)
                return; // not allocated;

            CUDA_CHECK_ERRORS(cudaFreeArray(vol.array_ptr));
            vol.array_ptr = nullptr;
        }
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
        
        if (format_desc.f == cudaChannelFormatKindFloat) {
            if (format_desc.x != 32) {
                FATAL() << "Unsupported format";
            }

            Type voxel_type = Type_Unknown;
            if (num_comp == 1) voxel_type = Type_Float;
            if (num_comp == 2) voxel_type = Type_Float2;
            if (num_comp == 3) voxel_type = Type_Float3;
            if (num_comp == 4) voxel_type = Type_Float4;

            return voxel_type;
        }
        else if (format_desc.f == cudaChannelFormatKindUnsigned) {
            if (format_desc.x == 8) {
                Type voxel_type = Type_Unknown;
                if (num_comp == 1) voxel_type = Type_UChar;
                if (num_comp == 2) voxel_type = Type_UChar2;
                if (num_comp == 3) voxel_type = Type_UChar3;
                if (num_comp == 4) voxel_type = Type_UChar4;
                return voxel_type;
            }
        }

        FATAL() << "Unsupported format";
        return Type_Unknown;
    }
}

GpuVolumeData::GpuVolumeData() :
    usage(gpu::Usage_PitchedPointer),
    array_ptr(nullptr)
{
    format_desc = {0};
    pitched_ptr = {0};
}
GpuVolumeData::~GpuVolumeData()
{
    release_gpu_volume(*this);
}

GpuVolume::GpuVolume()
{
    _size = {0};
    _origin = {0};
    _spacing = {0};
}
GpuVolume::GpuVolume(const dim3& size, Type voxel_type, gpu::Usage usage)
{
    _size = size;
    _origin = {0};
    _spacing = {0};
    _data = allocate_gpu_volume(size, voxel_type, usage);
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
    copy._origin = _origin;
    copy._spacing = _spacing;

    copy.copy_from(*this);

    return copy;
}
GpuVolume GpuVolume::clone_as(gpu::Usage usage) const
{
    GpuVolume copy(_size, voxel_type(), usage);
    copy._origin = _origin;
    copy._spacing = _spacing;

    copy.copy_from(*this);

    return copy;
}
void GpuVolume::copy_from(const GpuVolume& other)
{
    ASSERT(valid());
    ASSERT(other.valid());
    ASSERT(_size == other._size);
    ASSERT(voxel_type() == other.voxel_type());

    cudaMemcpy3DParms params = {0};
    params.kind = cudaMemcpyDeviceToDevice;

    // Extent width is defined in terms of elements if any cudaArray is present,
    //  otherwise in number of bytes (for pitched pointer)
    size_t per_voxel = 1;
    if (usage() == gpu::Usage_PitchedPointer && other.usage() == gpu::Usage_PitchedPointer) {
        per_voxel = (_data->format_desc.x + _data->format_desc.y + _data->format_desc.z + _data->format_desc.w) / 8;
    }

    if (other.usage() == gpu::Usage_PitchedPointer) {
        params.srcPtr = other.pitched_ptr();
    }
    else {
        params.srcArray = other.array_ptr();
    }
    
    if (usage() == gpu::Usage_PitchedPointer) {
        params.dstPtr = pitched_ptr();
    }
    else {
        params.dstArray = array_ptr();
    }

    params.extent = make_cudaExtent(
        _size.x * per_voxel,
        _size.y, 
        _size.z
    );
    
    CUDA_CHECK_ERRORS(cudaMemcpy3D(&params));
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

const float3& GpuVolume::origin() const
{
    return _origin;
}
const float3& GpuVolume::spacing() const
{
    return _spacing;
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
void GpuVolume::download(Volume& vol) const
{
    ASSERT(valid()); // Requires gpu memory to be allocated
    ASSERT(vol.valid()); // Requires cpu memory to be allocated as well

    // We also assume both volumes have same dimensions
    ASSERT(vol.size() == _size);

    // TODO: Validate format?

    cudaMemcpy3DParms params = {0};
    params.kind = cudaMemcpyDeviceToHost;
    params.dstPtr = make_cudaPitchedPtr(const_cast<void*>(vol.ptr()),
        _size.x * type_size(vol.voxel_type()), _size.x, _size.y);

    // Extent width is defined in terms of elements if any cudaArray is present,
    //  otherwise in number of bytes (for pitched pointer)
    size_t per_voxel = 1;
    if (usage() == gpu::Usage_PitchedPointer) {
        params.srcPtr = pitched_ptr();
        per_voxel = type_size(vol.voxel_type());
    }
    else {
        params.srcArray = array_ptr();
        per_voxel = 1;
    }

    params.extent = make_cudaExtent(_size.x * per_voxel, _size.y, _size.z);

    CUDA_CHECK_ERRORS(cudaMemcpy3D(&params));

    vol.set_origin(_origin);
    vol.set_spacing(_spacing);
}
void GpuVolume::upload(const Volume& vol)
{
    ASSERT(valid()); // Requires gpu memory to be allocated
    ASSERT(vol.valid()); // Requires cpu memory to be allocated as well

    // We also assume both volumes have same dimensions
    ASSERT(vol.size() == _size);

    // TODO: Validate format?

    cudaMemcpy3DParms params = {0};
    params.kind = cudaMemcpyHostToDevice;
    params.srcPtr = make_cudaPitchedPtr(const_cast<void*>(vol.ptr()),
        _size.x * type_size(vol.voxel_type()), _size.x, _size.y);

    
    // Extent width is defined in terms of elements if any cudaArray is present,
    //  otherwise in number of bytes (for pitched pointer)
    size_t per_voxel = 1;
    if (usage() == gpu::Usage_PitchedPointer) {
        params.dstPtr = pitched_ptr();
        per_voxel = type_size(vol.voxel_type());
    }
    else {
        params.dstArray = array_ptr();
        per_voxel = 1;
    }
    params.extent = make_cudaExtent(_size.x * per_voxel, _size.y, _size.z);
    
    CUDA_CHECK_ERRORS(cudaMemcpy3D(&params));

    set_origin(vol.origin());
    set_spacing(vol.spacing());
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
    copy._origin = _origin;
    copy._spacing = _spacing;

    copy.copy_from(*this);

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
    return _data->pitched_ptr;
}
void GpuVolume::allocate(const dim3& size, Type voxel_type, gpu::Usage usage)
{
    _size = size;
    _data = allocate_gpu_volume(size, voxel_type, usage);
}
} // namespace stk

#endif // STK_USE_CUDA
