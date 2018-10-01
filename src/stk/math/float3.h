#pragma once

#include "types.h"

#include <stk/cuda/cuda.h>
#include <cmath>

namespace stk
{
    inline CUDA_HOST_DEVICE float norm(const float3& v)
    {
        return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }
    inline CUDA_HOST_DEVICE float norm2(const float3& v)
    {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }
}

inline CUDA_HOST_DEVICE float3 operator+(const float3& l, const float3& r)
{
    return { l.x + r.x, l.y + r.y, l.z + r.z };
}
inline CUDA_HOST_DEVICE float3 operator-(const float3& l, const float3& r)
{
    return { l.x - r.x, l.y - r.y, l.z - r.z };
}

// element-wise multiplication
inline CUDA_HOST_DEVICE float3 operator*(const float3& l, const float3& r)
{
    return { l.x * r.x, l.y * r.y, l.z * r.z };
}
// element-wise division
inline CUDA_HOST_DEVICE float3 operator/(const float3& l, const float3& r)
{
    return { l.x / r.x, l.y / r.y, l.z / r.z };
}

inline CUDA_HOST_DEVICE float3 operator*(float l, const float3& r)
{
    return { r.x * l, r.y * l, r.z * l };
}
inline CUDA_HOST_DEVICE float3 operator*(const float3& l, float r)
{
    return { l.x * r, l.y * r, l.z * r };
}
inline CUDA_HOST_DEVICE float3 operator*(double l, const float3& r)
{
    return { float(r.x * l), float(r.y * l), float(r.z * l) };
}
inline CUDA_HOST_DEVICE float3 operator*(const float3& l, double r)
{
    return { float(l.x * r), float(l.y * r), float(l.z * r) };
}

inline CUDA_HOST_DEVICE float3 operator/(const float3& l, float r)
{
    return { l.x / r, l.y / r, l.z / r };
}
inline CUDA_HOST_DEVICE float3 operator/(const float3& l, double r)
{
    return { float(l.x / r), float(l.y / r), float(l.z / r) };
}
inline CUDA_HOST_DEVICE float3& operator+=(float3& l, const float3& r)
{
    l.x += r.x;
    l.y += r.y;
    l.z += r.z;
    return l;
}
inline CUDA_HOST_DEVICE float dot(float3 l, float3 r)
{
    return l.x * r.x + l.y * r.y + l.z * r.z;
}


