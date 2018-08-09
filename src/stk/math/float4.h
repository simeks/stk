#pragma once


inline __host__ __device__ float4 operator+(const float4& l, const float4& r)
{
    return { l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w };
}
inline __host__ __device__ float4 operator-(const float4& l, const float4& r)
{
    return { l.x - r.x, l.y - r.y, l.z - r.z, l.w - r.w };
}

// element-wise multiplication
inline __host__ __device__ float4 operator*(const float4& l, const float4& r)
{
    return { l.x * r.x, l.y * r.y, l.z * r.z, l.w * r.w };
}
// element-wise division
inline __host__ __device__ float4 operator/(const float4& l, const float4& r)
{
    return { l.x / r.x, l.y / r.y, l.z / r.z, l.w / r.w };
}

inline __host__ __device__ float4 operator*(float l, const float4& r)
{
    return { r.x * l, r.y * l, r.z * l, r.w * l };
}
inline __host__ __device__ float4 operator/(const float4& l, float r)
{
    return { l.x / r, l.y / r, l.z / r, l.w / r };
}


