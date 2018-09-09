#include <cstring>

#include "volume.h"
#include "stk/common/assert.h"

#ifdef STK_USE_CUDA
    #include <stk/cuda/cuda.h>
#endif

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
VolumeData::VolumeData() : data(NULL), size(0), flags(0)
{
}
VolumeData::VolumeData(size_t size, uint32_t flags) : size(size), flags(flags)
{
#ifdef STK_USE_CUDA
    if (flags & Usage_Pinned) {
        CUDA_CHECK_ERRORS(cudaHostAlloc(&data, size, cudaHostAllocDefault));
    }
    else if (flags & Usage_Mapped) {
        CUDA_CHECK_ERRORS(cudaHostAlloc(&data, size, cudaHostAllocMapped));
    }
    else if (flags & Usage_WriteCombined) {
        CUDA_CHECK_ERRORS(cudaHostAlloc(&data, size, cudaHostAllocWriteCombined));
    }
    else {
        data = (uint8_t*)malloc(size);
    }
#else
    ASSERT(flags == 0);
    data = (uint8_t*)malloc(size);
#endif

}
VolumeData::~VolumeData()
{
#ifdef STK_USE_CUDA
    if (data)
    {
        if (flags & (Usage_Pinned | Usage_Mapped | Usage_WriteCombined)) {
            CUDA_CHECK_ERRORS(cudaFreeHost(data));
        }
        else {
            free(data);
        }
    }
#else
    if (data)
        free(data);
#endif
}

Volume::Volume() : 
    _ptr(NULL),
    _strides{0,0,0},
    _voxel_type(Type_Unknown), 
    _contiguous(true)
{
    
    _origin = {0, 0, 0};
    _spacing = {1, 1, 1};
}
Volume::Volume(const dim3& size, Type voxel_type, const void* data, uint32_t flags) :
    _size(size),
    _voxel_type(voxel_type),
    _contiguous(true)
{
    _origin = {0, 0, 0};
    _spacing = {1, 1, 1};

    allocate(size, voxel_type, flags);
    if (data) {
        size_t num_bytes = _size.x * _size.y *
            _size.z * type_size(_voxel_type);

        memcpy(_ptr, data, num_bytes);
    }
}
Volume::Volume(const Volume& other, const Range& x, const Range& y, const Range& z) :
    _data(other._data),
    _ptr(other._ptr),
    _voxel_type(other._voxel_type),
    _origin(other._origin),
    _spacing(other._spacing)
{
    ASSERT(other.valid());
    ASSERT(x.begin <= x.end && y.begin <= y.end && z.begin <= z.end);
    ASSERT(0 <= x.begin && x.end <= (int)other._size.x);
    ASSERT(0 <= y.begin && y.end <= (int)other._size.y);
    ASSERT(0 <= z.begin && z.end <= (int)other._size.z);

    _strides[0] = other._strides[0];
    _strides[1] = other._strides[1];
    _strides[2] = other._strides[2];

    int nx = x.end - x.begin;
    int ny = y.end - y.begin;
    int nz = z.end - z.begin;

    uint8_t* ptr = reinterpret_cast<uint8_t*>(_ptr)
        + x.begin * _strides[0] + y.begin * _strides[1] + z.begin * _strides[2];
    
        // any offset in z axis does not break contiguity 
    _contiguous = true;
    if (nx != (int)other._size.x || ny != (int)other._size.y) {
        _contiguous = false;
    }

    _size = dim3{(uint32_t)nx, (uint32_t)ny, (uint32_t)nz};
    _ptr = ptr;
}
Volume::~Volume()
{
}
Volume Volume::clone() const
{
    Volume copy(_size, _voxel_type, nullptr, _data->flags);
    copy.copy_from(*this);
    return copy;
}
void Volume::copy_from(const Volume& other)
{
    ASSERT(_size == other._size);
    ASSERT(_voxel_type == other._voxel_type);

    if (is_contiguous() && other.is_contiguous()) {
        size_t num_bytes = _size.x * _size.y * 
            _size.z * type_size(_voxel_type);
        memcpy(_ptr, other._ptr, num_bytes);
    }
    else {
        size_t row_bytes = type_size(voxel_type()) * _size.x;

        for (int z = 0; z < (int)_size.z; ++z) {
            uint8_t* dst_row = reinterpret_cast<uint8_t*>(_ptr) + z * _strides[2];
            uint8_t* src_row = reinterpret_cast<uint8_t*>(other._ptr) + z * other._strides[2];

            for (int y = 0; y < (int)_size.y; ++y) {
                memcpy(dst_row, src_row, row_bytes);

                dst_row += _strides[1];
                src_row += other._strides[1];
            }
        }
    }
    
    _origin = other._origin;
    _spacing = other._spacing;
}
Volume Volume::as_type(Type type) const
{
    ASSERT(valid());
    ASSERT(type != Type_Unknown);
    // Non-contiguous not supported for now, this function is gonna be refactored anyway
    ASSERT(is_contiguous());
    if (_voxel_type == type)
        return *this;

    FATAL_IF(num_components(type) != num_components(_voxel_type)) <<
        "Cannot convert between voxel types with different number of components.";

    Volume dest(_size, type, nullptr, _data->flags);
    
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
const size_t* Volume::strides() const
{
    return _strides;
}
void Volume::copy_meta_from(const Volume& other)
{
    _origin = other._origin;
    _spacing = other._spacing;
}
bool Volume::is_contiguous() const
{
    return _contiguous;
}
Volume Volume::operator()(const Range& x, const Range& y, const Range& z)
{
    return Volume(*this, x, y, z);
}

Volume::Volume(const Volume& other) :
    _data(other._data),
    _ptr(other._ptr),
    _size(other._size),
    _voxel_type(other._voxel_type),
    _origin(other._origin),
    _spacing(other._spacing),
    _contiguous(other._contiguous)
{
    _strides[0] = other._strides[0];
    _strides[1] = other._strides[1];
    _strides[2] = other._strides[2];
}
Volume& Volume::operator=(const Volume& other)
{
    if (this != &other) {
        _data = other._data;
        _ptr = other._ptr;
        _strides[0] = other._strides[0];
        _strides[1] = other._strides[1];
        _strides[2] = other._strides[2];
        _size = other._size;
        _voxel_type = other._voxel_type;
        _origin = other._origin;
        _spacing = other._spacing;
        _contiguous = other._contiguous;
    }
    return *this;
}
void Volume::allocate(const dim3& size, Type voxel_type, uint32_t flags)
{
    ASSERT(voxel_type != Type_Unknown);

    _size = size;
    _voxel_type = voxel_type;
    _origin = { 0, 0, 0 };
    _spacing = { 1, 1, 1 };

    size_t num_bytes = _size.x * _size.y *
        _size.z * type_size(_voxel_type);

    _data = std::make_shared<VolumeData>(num_bytes, flags);
    _ptr = _data->data;
    _strides[0] = type_size(_voxel_type);
    _strides[1] = _strides[0] * _size.x;
    _strides[2] = _strides[1] * _size.y;

    _contiguous = true;
}
void Volume::release()
{
    _data.reset();
    _ptr = NULL;
    _size = { 0, 0, 0 };
    _strides[0] = 0;
    _strides[1] = 0;
    _strides[2] = 0;
    _origin = { 0, 0, 0 };
    _spacing = { 1, 1, 1 };
}
std::vector<std::string> Volume::get_metadata_keys(void) const
{
    std::vector<std::string> keys;
    keys.reserve(_metadata.size());
    for (auto const& [k, _] : _metadata) {
        keys.push_back(k);
    }
    return keys;
}
std::any Volume::get_metadata(const std::string& key) const
{
    try {
        return _metadata.at(key);
    }
    catch (const std::out_of_range&) {
        FATAL() << "Metadata '" << &key << "' not found";
    }
}
void Volume::set_metadata(const std::string& key, const std::any& value)
{
    _metadata.emplace(key, value);
}
} // namespace stk

