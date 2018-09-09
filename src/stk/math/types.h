#pragma once

#include <stk/common/platform.h>

#include <algorithm>
#include <initializer_list>
#include <sstream>

#ifdef STK_USE_CUDA
// float2, float3, etc...
#include <vector_types.h>

#else // STK_USE_CUDA

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
    uint32_t x, y;
};
struct uint3
{
    uint32_t x, y, z;
};
struct STK_ALIGN(16) uint4
{
    uint32_t x, y, z, w;
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


#endif // STK_USE_CUDA


template<typename T, unsigned int rows_, unsigned int cols_>
struct matrix
{
    T _data[rows_][cols_];

    static constexpr unsigned int rows = rows_;
    static constexpr unsigned int cols = cols_;

    T* operator[](const unsigned int i) {
        return _data[i];
    }

    const T* operator[](const unsigned int i) const {
        return _data[i];
    }

    void diagonal(const std::initializer_list<T> d) {
        ASSERT(d.size() == std::min(rows_, cols));
        std::fill(&_data[0][0], &_data[0][0] + rows_ * cols_, T(0));
        int i = 0;
        for (const auto& x : d) {
            _data[i][i] = x;
            i++;
        }
    }
};

using Matrix3x3f = matrix<float, 3, 3>;


// char

inline bool operator==(char2 l, char2 r)
{
    return (l.x == r.x && l.y == r.y);
}
inline bool operator!=(char2 l, char2 r)
{
    return !operator==(l, r);
}

inline bool operator==(char3 l, char3 r)
{
    return (l.x == r.x && l.y == r.y && l.z == r.z);
}
inline bool operator!=(char3 l, char3 r)
{
    return !operator==(l, r);
}

inline bool operator==(char4 l, char4 r)
{
    return (l.x == r.x && l.y == r.y && l.z == r.z && l.w == r.w);
}
inline bool operator!=(char4 l, char4 r)
{
    return !operator==(l, r);
}

// uchar

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
    return (l.x == r.x && l.y == r.y && l.z == r.z);
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

// short

inline bool operator==(short2 l, short2 r)
{
    return (l.x == r.x && l.y == r.y);
}
inline bool operator!=(short2 l, short2 r)
{
    return !operator==(l, r);
}

inline bool operator==(short3 l, short3 r)
{
    return (l.x == r.x && l.y == r.y && l.z == r.z);
}
inline bool operator!=(short3 l, short3 r)
{
    return !operator==(l, r);
}

inline bool operator==(short4 l, short4 r)
{
    return (l.x == r.x && l.y == r.y && l.z == r.z && l.w == r.w);
}
inline bool operator!=(short4 l, short4 r)
{
    return !operator==(l, r);
}

// ushort

inline bool operator==(ushort2 l, ushort2 r)
{
    return (l.x == r.x && l.y == r.y);
}
inline bool operator!=(ushort2 l, ushort2 r)
{
    return !operator==(l, r);
}

inline bool operator==(ushort3 l, ushort3 r)
{
    return (l.x == r.x && l.y == r.y && l.z == r.z);
}
inline bool operator!=(ushort3 l, ushort3 r)
{
    return !operator==(l, r);
}

inline bool operator==(ushort4 l, ushort4 r)
{
    return (l.x == r.x && l.y == r.y && l.z == r.z && l.w == r.w);
}
inline bool operator!=(ushort4 l, ushort4 r)
{
    return !operator==(l, r);
}

// int

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
    return (l.x == r.x && l.y == r.y && l.z == r.z);
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

// uint

inline bool operator==(uint2 l, uint2 r)
{
    return (l.x == r.x && l.y == r.y);
}
inline bool operator!=(uint2 l, uint2 r)
{
    return !operator==(l, r);
}

inline bool operator==(uint3 l, uint3 r)
{
    return (l.x == r.x && l.y == r.y && l.z == r.z);
}
inline bool operator!=(uint3 l, uint3 r)
{
    return !operator==(l, r);
}

inline bool operator==(uint4 l, uint4 r)
{
    return (l.x == r.x && l.y == r.y && l.z == r.z && l.w == r.w);
}
inline bool operator!=(uint4 l, uint4 r)
{
    return !operator==(l, r);
}




// Overloads for convenient logging of types

inline std::ostream& operator<<(std::ostream& s, const char2& v)
{
    s << '(' << int(v.x) << ' ' << int(v.y) << ')';
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const char3& v)
{
    s << '(' << int(v.x) << ' ' << int(v.y) << ' ' << int(v.z) << ')';
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const char4& v)
{
    s << '(' << int(v.x) << ' ' << int(v.y) << ' ' << int(v.z) << ' ' << int(v.w) << ')';
    return s;
}

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

inline std::ostream& operator<<(std::ostream& s, const short2& v)
{
    s << '(' << v.x << ' ' << v.y << ')';
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const short3& v)
{
    s << '(' << v.x << ' ' << v.y << ' ' << v.z << ')';
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const short4& v)
{
    s << '(' << v.x << ' ' << v.y << ' ' << v.z << ' ' << v.w << ')';
    return s;
}

inline std::ostream& operator<<(std::ostream& s, const ushort2& v)
{
    s << '(' << v.x << ' ' << v.y << ')';
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const ushort3& v)
{
    s << '(' << v.x << ' ' << v.y << ' ' << v.z << ')';
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const ushort4& v)
{
    s << '(' << v.x << ' ' << v.y << ' ' << v.z << ' ' << v.w << ')';
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

inline std::ostream& operator<<(std::ostream& s, const uint2& v)
{
    s << '(' << v.x << ' ' << v.y << ')';
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const uint3& v)
{
    s << '(' << v.x << ' ' << v.y << ' ' << v.z << ')';
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const uint4& v)
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
