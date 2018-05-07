#pragma once

#include <stk/common/platform.h>

#include <sstream>

#ifdef STK_USE_CUDA
// float2, float3, etc...
#include <vector_types.h>

#else

#include <stdint.h>

// Defines some types otherwise defined by CUDA SDK.

struct STK_ALIGN(2) char2
{
    char x, y;
};
struct char3
{
    char x, y, z;
};
struct STK_ALIGN(4) char4
{
    char x, y, z, w;
};


struct STK_ALIGN(2) uchar2
{
    uint8_t x, y;
};
struct uchar3
{
    uint8_t x, y, z;
};
struct STK_ALIGN(4) uchar4
{
    uint8_t x, y, z, w;
};


struct STK_ALIGN(4) short2
{
    int16_t x, y;
};
struct short3
{
    int16_t x, y, z;
};
struct STK_ALIGN(8) short4
{
    int16_t x, y, z, w;
};


struct STK_ALIGN(4) ushort2
{
    uint16_t x, y;
};
struct ushort3
{
    uint16_t x, y, z;
};
struct STK_ALIGN(8) ushort4
{
    uint16_t x, y, z, w;
};


struct STK_ALIGN(8) int2
{
    int x, y;
};
struct int3
{
    int x, y, z;
};
struct STK_ALIGN(16) int4
{
    int x, y, z, w;
};


struct STK_ALIGN(8) uint2
{
    int x, y;
};
struct uint3
{
    int x, y, z;
};
struct STK_ALIGN(16) uint4
{
    int x, y, z, w;
};


struct STK_ALIGN(8) float2
{
    float x, y;
};
struct float3
{
    float x, y, z;
};
struct STK_ALIGN(16) float4
{
    float x, y, z, w;
};


struct STK_ALIGN(16) double2
{
    double x, y;
};
struct double3
{
    double x, y, z;
};
struct STK_ALIGN(16) double4
{
    double x, y, z, w;
};


#endif

inline bool operator==(uchar2 l, uchar2 r)
{
    return (l.x == r.x && l.y == r.y);
}
inline bool operator!=(uchar2 l, uchar2 r)
{
    return !operator==(l, r);
}

inline bool operator==(uchar3 l, uchar3 r)
{
    return (l.x == r.x && l.y == r.y && l.y == r.y);
}
inline bool operator!=(uchar3 l, uchar3 r)
{
    return !operator==(l, r);
}

inline bool operator==(uchar4 l, uchar4 r)
{
    return (l.x == r.x && l.y == r.y && l.z == r.z && l.w == r.w);
}
inline bool operator!=(uchar4 l, uchar4 r)
{
    return !operator==(l, r);
}

inline bool operator==(int2 l, int2 r)
{
    return (l.x == r.x && l.y == r.y);
}
inline bool operator!=(int2 l, int2 r)
{
    return !operator==(l, r);
}

inline bool operator==(int3 l, int3 r)
{
    return (l.x == r.x && l.y == r.y && l.y == r.y);
}
inline bool operator!=(int3 l, int3 r)
{
    return !operator==(l, r);
}

inline bool operator==(int4 l, int4 r)
{
    return (l.x == r.x && l.y == r.y && l.z == r.z && l.w == r.w);
}
inline bool operator!=(int4 l, int4 r)
{
    return !operator==(l, r);
}

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