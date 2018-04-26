#pragma once

#include <sstream>

#ifdef STK_USE_CUDA
// float2, float3, etc...
#include <vector_types.h>

#else

#include <stdint.h>

// Defines some types otherwise defined by CUDA SDK.
// However, not the same guarantees on alignmnet.

struct uchar2
{
    uint8_t x, y;
};

struct uchar3
{
    uint8_t x, y, z;
};

struct uchar4
{
    uint8_t x, y, z, w;
};

struct int2
{
    int x, y;
};

struct int3
{
    int x, y, z;
};

struct int4
{
    int x, y, z, w;
};

struct float2
{
    float x, y;
};

struct float3
{
    float x, y, z;
};

struct float4
{
    float x, y, z, w;
};

struct double2
{
    double x, y;
};

struct double3
{
    double x, y, z;
};

struct double4
{
    double x, y, z, w;
};

#endif

// Overloads for convenient logging of types

inline std::ostream& operator<<(std::ostream& s, const uchar2& v)
{
    s << '(' << int(v.x) << ' ' << int(v.y) << ')';
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const uchar3& v)
{
    s << '(' << int(v.x) << ' ' << int(v.y) << ' ' << int(v.z) << ')';
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const uchar4& v)
{
    s << '(' << int(v.x) << ' ' << int(v.y) << ' ' << int(v.z) << ' ' << int(v.w) << ')';
    return s;
}

inline std::ostream& operator<<(std::ostream& s, const int2& v)
{
    s << '(' << v.x << ' ' << v.y << ')';
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const int3& v)
{
    s << '(' << v.x << ' ' << v.y << ' ' << v.z << ')';
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const int4& v)
{
    s << '(' << v.x << ' ' << v.y << ' ' << v.z << ' ' << v.w << ')';
    return s;
}

inline std::ostream& operator<<(std::ostream& s, const float2& v)
{
    s << '(' << v.x << ' ' << v.y << ')';
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const float3& v)
{
    s << '(' << v.x << ' ' << v.y << ' ' << v.z << ')';
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const float4& v)
{
    s << '(' << v.x << ' ' << v.y << ' ' << v.z << ' ' << v.w << ')';
    return s;
}

inline std::ostream& operator<<(std::ostream& s, const double2& v)
{
    s << '(' << v.x << ' ' << v.y << ')';
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const double3& v)
{
    s << '(' << v.x << ' ' << v.y << ' ' << v.z << ')';
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const double4& v)
{
    s << '(' << v.x << ' ' << v.y << ' ' << v.z << ' ' << v.w << ')';
    return s;
}