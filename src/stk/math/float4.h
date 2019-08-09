#pragma once

#include <stk/cuda/cuda.h>

namespace stk
{
    inline CUDA_HOST_DEVICE float norm(const float4& v)
    {
        return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    }
    inline CUDA_HOST_DEVICE float norm2(const float4& v)
    {
        return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
}

inline CUDA_HOST_DEVICE float4 operator+(const float4& l, const float4& r)
{
    return { l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w };
}
inline CUDA_HOST_DEVICE float4 operator-(const float4& l, const float4& r)
{
    return { l.x - r.x, l.y - r.y, l.z - r.z, l.w - r.w };
}

// element-wise multiplication
inline CUDA_HOST_DEVICE float4 operator*(const float4& l, const float4& r)
{
    return { l.x * r.x, l.y * r.y, l.z * r.z, l.w * r.w };
}
// element-wise division
inline CUDA_HOST_DEVICE float4 operator/(const float4& l, const float4& r)
{
    return { l.x / r.x, l.y / r.y, l.z / r.z, l.w / r.w };
}

inline CUDA_HOST_DEVICE float4 operator*(float l, const float4& r)
{
    return { r.x * l, r.y * l, r.z * l, r.w * l };
}
inline CUDA_HOST_DEVICE float4 operator*(const float4& l, float r)
{
    return { l.x * r, l.y * r, l.z * r, l.w * r };
}
inline CUDA_HOST_DEVICE float4 operator/(const float4& l, float r)
{
    return { l.x / r, l.y / r, l.z / r, l.w / r };
}

inline CUDA_HOST_DEVICE float4& operator+=(float4& l, const float4& r)
{
    l.x += r.x;
    l.y += r.y;
    l.z += r.z;
    l.w += r.w;
    return l;
}
inline CUDA_HOST_DEVICE float dot(float4 l, float4 r)
{
    return l.x * r.x + l.y * r.y + l.z * r.z + l.w * r.w;
}


