#ifdef STK_USE_CUDA
    #include <cuda_runtime.h>

    #include "gpu_volume.h"
    #include "helper_cuda.h"
#endif

#include "volume.h"
#include "stk/common/assert.h"

namespace
{
    typedef void(*ConverterFn)(void*, void*, size_t num);

    template<typename TSrc, typename TDest>
    void convert_voxels(void* src, void* dest, size_t num)
    {
        // TODO:
        // Very rough conversion between voxel formats,
        // works fine for double <-> float, should suffice for now

        for (size_t i = 0; i < num; ++i) {
            ((TDest*)dest)[i] = TDest(((TSrc*)src)[i]);
        }
    }
}

namespace stk
{
VolumeData::VolumeData() : data(NULL), size(0)
{
}
VolumeData::VolumeData(size_t size) : size(size)
{
    data = (uint8_t*)malloc(size);
}
VolumeData::~VolumeData()
{
    if (data)
        free(data);
}

Volume::Volume() : _ptr(NULL), _stride(0), _voxel_type(Type_Unknown)
{
    _origin = {0, 0, 0};
    _spacing = {1, 1, 1};
}
Volume::Volume(const dim3& size, Type voxel_type, void* data) :
    _size(size),
    _voxel_type(voxel_type)
{
    _origin = {0, 0, 0};
    _spacing = {1, 1, 1};

    allocate(size, voxel_type);
    if (data) {
        size_t num_bytes = _size.x * _size.y *
            _size.z * type_size(_voxel_type);

        memcpy(_ptr, data, num_bytes);
    }
}
Volume::~Volume()
{
}
Volume Volume::clone() const
{
    Volume copy(_size, _voxel_type);
    copy._origin = _origin;
    copy._spacing = _spacing;

    size_t num_bytes = _size.x * _size.y * 
        _size.z * type_size(_voxel_type);
    
    memcpy(copy._ptr, _ptr, num_bytes);

    return copy;
}
void Volume::copy_from(const Volume& other)
{
    ASSERT(_size == other._size);
    ASSERT(_voxel_type == other._voxel_type);

    size_t num_bytes = _size.x * _size.y * 
        _size.z * type_size(_voxel_type);

    memcpy(_ptr, other._ptr, num_bytes);
    
    _origin = other._origin;
    _spacing = other._spacing;
}
Volume Volume::as_type(Type type) const
{
    if (_voxel_type == type)
        return *this;

    FATAL_IF(num_components(type) != num_components(_voxel_type)) <<
        "Cannot convert between voxel types with different number of components.";

    Volume dest(_size, type);
    
    Type src_type = base_type(_voxel_type);
    Type dest_type = base_type(type);

    size_t num = _size.x * _size.y * _size.z * num_components(type);
    if (src_type == Type_Float && dest_type == Type_Double)
        convert_voxels<float, double>(_ptr, dest._ptr, num);
    else if (src_type == Type_Double && dest_type == Type_Float)
        convert_voxels<double, float>(_ptr, dest._ptr, num);
    else
        NOT_IMPLEMENTED() << "Conversion from " << as_string(_voxel_type) 
                          << " to " << as_string(type) << " not supported";

    dest._origin = _origin;
    dest._spacing = _spacing;

    return dest;
}
bool Volume::valid() const
{
    return _ptr != NULL;
}
void* Volume::ptr()
{
    DASSERT(_ptr);
    DASSERT(_data->data);
    DASSERT(_data->size);
    return _ptr;
}
void const* Volume::ptr() const
{
    DASSERT(_ptr);
    DASSERT(_data->data);
    DASSERT(_data->size);
    return _ptr;
}
Type Volume::voxel_type() const
{
    return _voxel_type;
}
const dim3& Volume::size() const
{
    return _size;
}
void Volume::set_origin(const float3& origin)
{
    _origin = origin;
}
void Volume::set_spacing(const float3& spacing)
{
    _spacing = spacing;
}
const float3& Volume::origin() const
{
    return _origin;
}
const float3& Volume::spacing() const
{
    return _spacing;
}

Volume::Volume(const Volume& other) :
    _data(other._data),
    _ptr(other._ptr),
    _stride(other._stride),
    _size(other._size),
    _voxel_type(other._voxel_type),
    _origin(other._origin),
    _spacing(other._spacing)
{
}
Volume& Volume::operator=(const Volume& other)
{
    if (this != &other) {
        _data = other._data;
        _ptr = other._ptr;
        _stride = other._stride;
        _size = other._size;
        _voxel_type = other._voxel_type;
        _origin = other._origin;
        _spacing = other._spacing;
    }
    return *this;
}
void Volume::allocate(const dim3& size, Type voxel_type)
{
    ASSERT(voxel_type != Type_Unknown);

    _size = size;
    _voxel_type = voxel_type;
    _origin = { 0, 0, 0 };
    _spacing = { 1, 1, 1 };

    size_t num_bytes = _size.x * _size.y *
        _size.z * type_size(_voxel_type);

    _data = std::make_shared<VolumeData>(num_bytes);
    _ptr = _data->data;
    _stride = type_size(_voxel_type) * _size.x;
}
void Volume::release()
{
    _data.reset();
    _ptr = NULL;
    _size = { 0, 0, 0 };
    _stride = 0;
    _origin = { 0, 0, 0 };
    _spacing = { 1, 1, 1 };
}

#ifdef STK_USE_CUDA
Volume::Volume(const GpuVolume& gpu_volume)
{
    allocate(gpu_volume.size, gpu::voxel_type(gpu_volume));
    download(gpu_volume);
}
GpuVolume Volume::upload() const
{
    GpuVolume vol = gpu::allocate_volume(_voxel_type, _size);
    upload(vol);
    return vol;
}
void Volume::upload(const GpuVolume& gpu_volume) const
{
    assert(gpu_volume.ptr != NULL); // Requires gpu memory to be allocated
    assert(valid()); // Requires cpu memory to be allocated as well

    // We also assume both volumes have same dimensions
    assert( gpu_volume.size.x == _size.x &&
            gpu_volume.size.y == _size.y &&
            gpu_volume.size.z == _size.z);

    // TODO: Validate format?

    cudaMemcpy3DParms params = { 0 };
    params.srcPtr = make_cudaPitchedPtr(_ptr, _size.x * voxel::size(_voxel_type), _size.x, _size.y);
    params.dstArray = gpu_volume.ptr;
    params.extent = { gpu_volume.size.x, gpu_volume.size.y, gpu_volume.size.z };
    params.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&params));
}
void Volume::download(const GpuVolume& gpu_volume)
{
    assert(gpu_volume.ptr != NULL); // Requires gpu memory to be allocated
    assert(valid()); // Requires cpu memory to be allocated as well

    // We also assume both volumes have same dimensions
    assert( gpu_volume.size.x == _size.x &&
            gpu_volume.size.y == _size.y &&
            gpu_volume.size.z == _size.z);

    // TODO: Validate format?

    cudaMemcpy3DParms params = { 0 };
    params.srcArray = gpu_volume.ptr;
    params.dstPtr = make_cudaPitchedPtr(_ptr, _size.x * voxel::size(_voxel_type), _size.x, _size.y);
    params.extent = { gpu_volume.size.x, gpu_volume.size.y, gpu_volume.size.z };
    params.kind = cudaMemcpyDeviceToHost;
    checkCudaErrors(cudaMemcpy3D(&params));
}

#endif // STK_USE_CUDA

} // namespace stk

