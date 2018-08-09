#pragma once

#include <stk/cuda/cuda.h>

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
inline CUDA_HOST_DEVICE float4 operator/(const float4& l, float r)
{
    return { l.x / r, l.y / r, l.z / r, l.w / r };
}


