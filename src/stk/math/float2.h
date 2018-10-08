#pragma once

#include <stk/cuda/cuda.h>

#include "types.h"

#include <cmath>

namespace stk
{
    inline float norm(const float2& v)
    {
        return std::sqrt(v.x * v.x + v.y * v.y);
    }
    inline float norm2(const float2& v)
    {
        return v.x * v.x + v.y * v.y;
    }
}

inline CUDA_HOST_DEVICE float2 operator+(const float2& l, const float2& r)
{
    return { l.x + r.x, l.y + r.y };
}
inline CUDA_HOST_DEVICE float2 operator-(const float2& l, const float2& r)
{
    return { l.x - r.x, l.y - r.y };
}

// element-wise multiplication
inline CUDA_HOST_DEVICE float2 operator*(const float2& l, const float2& r)
{
    return { l.x * r.x, l.y * r.y };
}
// element-wise division
inline CUDA_HOST_DEVICE float2 operator/(const float2& l, const float2& r)
{
    return { l.x / r.x, l.y / r.y };
}

inline CUDA_HOST_DEVICE float2 operator*(float l, const float2& r)
{
    return { r.x * l, r.y * l };
}
inline CUDA_HOST_DEVICE float2 operator*(const float2& l, float r)
{
    return { l.x * r, l.y * r };
}
inline CUDA_HOST_DEVICE float2 operator*(double l, const float2& r)
{
    return { float(r.x * l), float(r.y * l) };
}
inline CUDA_HOST_DEVICE float2 operator*(const float2& l, double r)
{
    return { float(l.x * r), float(l.y * r) };
}

inline CUDA_HOST_DEVICE float2 operator/(const float2& l, float r)
{
    return { l.x / r, l.y / r };
}
inline CUDA_HOST_DEVICE float2 operator/(const float2& l, double r)
{
    return { float(l.x / r), float(l.y / r) };
}
inline CUDA_HOST_DEVICE float2& operator+=(float2& l, const float2& r)
{
    l.x += r.x;
    l.y += r.y;
    return l;
}
inline CUDA_HOST_DEVICE float dot(float2 l, float2 r)
{
    return l.x * r.x + l.y * r.y;
}


