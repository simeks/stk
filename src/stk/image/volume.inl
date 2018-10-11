#pragma once

namespace {

/*!
 * \brief Mirror an index ranging from 0 to n-1.
 */
static inline int mirror(int x, const int n)
{
    while (x < 0 || x >= n) {
        if (x < 0) {
            x = -x;
        }
        if (x >= n) {
            x = 2 * (n - 1) - x;
        }
    }
    return x;
}

} // namespace

namespace stk
{
template<typename T>
VolumeHelper<T>::VolumeHelper()
{
    _voxel_type = type_id<T>::id();
}
template<typename T>
VolumeHelper<T>::VolumeHelper(const Volume& other) : Volume()
{
    *this = other; // Operator performs conversion (if needed)
}
template<typename T>
VolumeHelper<T>::VolumeHelper(const dim3& size) : Volume(size, type_id<T>::id())
{
}
template<typename T>
VolumeHelper<T>::VolumeHelper(const dim3& size, const T& value) :
    Volume(size, type_id<T>::id())
{
    fill(value);
}
template<typename T>
VolumeHelper<T>::VolumeHelper(const dim3& size, T* value) :
    Volume(size, type_id<T>::id(), value)
{
}
template<typename T>
VolumeHelper<T>::VolumeHelper(
    const VolumeHelper<T>& other,
    const Range& x,
    const Range& y,
    const Range& z) : Volume(other, x, y, z)
{
}
template<typename T>
VolumeHelper<T>::~VolumeHelper()
{
}
template<typename T>
void VolumeHelper<T>::allocate(const dim3& size)
{
    Volume::allocate(size, type_id<T>::id());
}
template<typename T>
void VolumeHelper<T>::fill(const T& value)
{
    for (uint32_t z = 0; z < _size.z; ++z) {
        for (uint32_t y = 0; y < _size.y; ++y) {
            // x axis should always be contiguous
            T* begin = (T*)(((uint8_t*)_ptr) + (z * _strides[2] + y * _strides[1]));
            T* end = begin + _size.x;
            std::fill(begin, end, value);
        }
    }
}
template<typename T>
T VolumeHelper<T>::at(int x, int y, int z, BorderMode border_mode) const
{
    // PERF: 15% faster than ceilf/floorf
    #define FAST_CEIL(x_) ((int)x_ + (x_ > (int)x_))
    #define FAST_FLOOR(x_) ((int)x_ - (x_ < (int)x_))

    if (border_mode == Border_Constant)
    {
        if (x < 0 || FAST_CEIL(x) >= int(_size.x) ||
            y < 0 || FAST_CEIL(y) >= int(_size.y) ||
            z < 0 || FAST_CEIL(z) >= int(_size.z))
        {
            return T{};
        }
    }
    else if (border_mode == Border_Replicate)
    {
        x = std::max(x, 0);
        x = std::min(x, int(_size.x - 1));
        y = std::max(y, 0);
        y = std::min(y, int(_size.y - 1));
        z = std::max(z, 0);
        z = std::min(z, int(_size.z - 1));
    }
    else if (border_mode == Border_Mirror)
    {
        x = ::mirror(x, _size.x);
        y = ::mirror(y, _size.y);
        z = ::mirror(z, _size.z);
    }
    else if (border_mode == Border_Cyclic)
    {
        x = x % _size.x;
        y = y % _size.y;
        z = z % _size.z;
    }

    return *((T const*)(((uint8_t*)_ptr) + offset(x, y, z)));

    #undef FAST_CEIL
    #undef FAST_FLOOR
}
template<typename T>
T VolumeHelper<T>::at(int3 p, BorderMode border_mode) const
{
    return at(p.x, p.y, p.z, border_mode);
}
template<typename T>
inline T VolumeHelper<T>::linear_at(float x, float y, float z, BorderMode border_mode) const
{
    if (border_mode == Border_Constant) {
        if (x < 0 || int(x) >= int(_size.x) ||
            y < 0 || int(y) >= int(_size.y) ||
            z < 0 || int(z) >= int(_size.z)) {
            return T{};
        }
    }
    else if (border_mode == Border_Replicate) {
        x = std::max(x, 0.0f);
        x = std::min(x, float(_size.x - 1));
        y = std::max(y, 0.0f);
        y = std::min(y, float(_size.y - 1));
        z = std::max(z, 0.0f);
        z = std::min(z, float(_size.z - 1));
    }

    // Floor
    int x1 = int(x);
    int y1 = int(y);
    int z1 = int(z);

    // Ceil
    int x2 = std::min<int>(x1+1, int(_size.x-1));
    int y2 = std::min<int>(y1+1, int(_size.y-1));
    int z2 = std::min<int>(z1+1, int(_size.z-1));

    float xt = x - x1;
    float yt = y - y1;
    float zt = z - z1;

    return T(
        (1 - zt) *
            (
                (1 - yt) *
                (
                    (1 - xt) * operator()(x1, y1, z1) + // s1
                    (xt) * operator()(x2, y1, z1) // s2
                ) +

                (yt) *
                (
                    (1 - xt) * operator()(x1, y2, z1) + // s3
                    (xt) * operator()(x2, y2, z1) // s4
                )
            ) +
        (zt) *
            (
                (1 - yt)*
                (
                    (1 - xt)*operator()(x1, y1, z2) + // s5
                    (xt)*operator()(x2, y1, z2) // s6
                ) +

                (yt)*
                (
                    (1 - xt)*operator()(x1, y2, z2) + // s7
                    (xt)*operator()(x2, y2, z2) // s8
                )
            )
        );
}

template<typename T>
T VolumeHelper<T>::linear_at(float3 p, BorderMode border_mode) const
{
    return linear_at(p.x, p.y, p.z, border_mode);
}
template<typename T>
inline T VolumeHelper<T>::linear_at_point(float x, float y, float z, BorderMode border_mode) const
{
    const float3 p = point2index(float3({x, y, z}));
    return linear_at(p.x, p.y, p.z, border_mode);
}
template<typename T>
T VolumeHelper<T>::linear_at_point(float3 p, BorderMode border_mode) const
{
    p = point2index(p);
    return linear_at(p.x, p.y, p.z, border_mode);
}
template<typename T>
VolumeHelper<T>& VolumeHelper<T>::operator=(const VolumeHelper& other)
{
    ASSERT(type_id<T>::id() == other.voxel_type()); // Sanity check
    Volume::operator=(other);
    return *this;
}
template<typename T>
VolumeHelper<T>& VolumeHelper<T>::operator=(const Volume& other)
{
    if (static_cast<enum stk::Type>(type_id<T>::id()) == other.voxel_type()) {
        Volume::operator=(other);
        return *this;
    }
    if (!other.valid()) {
        release();
        return *this;
    }

    *this = other.as_type(type_id<T>::id());
    return *this;
}

template<typename T>
const T& VolumeHelper<T>::operator()(int x, int y, int z) const
{
    return *((T const*)(((uint8_t*)_ptr) + offset(x, y, z)));
}
template<typename T>
T& VolumeHelper<T>::operator()(int x, int y, int z)
{
    return *((T*)(((uint8_t*)_ptr) + offset(x, y, z)));
}
template<typename T>
const T& VolumeHelper<T>::operator()(const int3& p) const
{
    return operator()(p.x, p.y, p.z);
}
template<typename T>
T& VolumeHelper<T>::operator()(const int3& p)
{
    return operator()(p.x, p.y, p.z);
}
template<typename T>
VolumeHelper<T> VolumeHelper<T>::operator()(const Range& x, const Range& y, const Range& z)
{
    return Volume::operator()(x,y,z);
}
template<typename T>
inline size_t VolumeHelper<T>::offset(int x, int y, int z) const
{
    DASSERT(x < int(_size.x));
    DASSERT(y < int(_size.y));
    DASSERT(z < int(_size.z));
    return z * _strides[2] + y * _strides[1] + x * _strides[0];
}

template<typename T>
void find_min_max(const VolumeHelper<T>& vol, T& min, T& max)
{
    ASSERT(num_components(vol.voxel_type()) == 1);

    min = std::numeric_limits<T>::max();
    max = std::numeric_limits<T>::lowest();

    dim3 size = vol.size();
    for (uint32_t z = 0; z < size.z; ++z)
    {
        for (uint32_t y = 0; y < size.y; ++y)
        {
            for (uint32_t x = 0; x < size.x; ++x)
            {
                min = std::min<T>(min, vol(x, y, z));
                max = std::max<T>(max, vol(x, y, z));
            }
        }
    }
}
} // namespace stk
