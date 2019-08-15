#pragma once

#include <stk/common/assert.h>
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


struct Matrix3x3f
{
    enum : unsigned int {
        rows = 3,
        cols = 3
    };

    float3 _rows[rows];

    float3& operator[](const unsigned int i) {
        return _rows[i];
    }

    const float3& operator[](const unsigned int i) const {
        return _rows[i];
    }

    const float3 column(const unsigned i) const {
        ASSERT(i < cols);
        return float3({(*this)(0, i), (*this)(1, i), (*this)(2, i)});
    }

    const float& operator()(const unsigned int r, const unsigned int c) const {
        ASSERT(c < cols && r < rows);
        return *(reinterpret_cast<const float*>(_rows + r) + c);
    }

    float& operator()(const unsigned int r, const unsigned int c) {
        return const_cast<float&>(static_cast<const Matrix3x3f*>(this)->operator()(r, c));
    }

    float* data(void) {
        return reinterpret_cast<float*>(&_rows[0]);
    }

    const float* data(void) const {
        return reinterpret_cast<const float*>(&_rows[0]);
    }

    void set(const float *data) {
        std::copy(data, data + rows*cols, (float*) _rows);
    }

    void set(const std::initializer_list<float> data) {
        std::copy(data.begin(), data.end(), (float*) _rows);
    }

    void diagonal(const std::initializer_list<float> d) {
        ASSERT(d.size() == std::min(rows, cols));
        std::fill(data(), data() + rows * cols, float(0));
        int i = 0;
        for (const auto& x : d) {
            (*this)(i, i) = x;
            i++;
        }
    }

    float determinant(void) const {
        return (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1)) -
               (*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0)) +
               (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
    }

    Matrix3x3f inverse(void) const {
        const float det = determinant();
        if (std::abs(det) < std::numeric_limits<float>::epsilon()) {
            FATAL() << "The matrix is not invertible";
        }
        // NOTE: we cannot assume that the direction matrix is orthogonal
        const float inv_det = 1.0f / det;
        Matrix3x3f res;
        res(0, 0) = inv_det * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1));
        res(0, 1) = inv_det * ((*this)(0, 2) * (*this)(2, 1) - (*this)(0, 1) * (*this)(2, 2));
        res(0, 2) = inv_det * ((*this)(0, 1) * (*this)(1, 2) - (*this)(0, 2) * (*this)(1, 1));
        res(1, 0) = inv_det * ((*this)(1, 2) * (*this)(2, 0) - (*this)(1, 0) * (*this)(2, 2));
        res(1, 1) = inv_det * ((*this)(0, 0) * (*this)(2, 2) - (*this)(0, 2) * (*this)(2, 0));
        res(1, 2) = inv_det * ((*this)(0, 2) * (*this)(1, 0) - (*this)(0, 0) * (*this)(1, 2));
        res(2, 0) = inv_det * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
        res(2, 1) = inv_det * ((*this)(0, 1) * (*this)(2, 0) - (*this)(0, 0) * (*this)(2, 1));
        res(2, 2) = inv_det * ((*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0));
        return res;
    }
};


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

inline std::ostream& operator<<(std::ostream& s, const Matrix3x3f& m)
{
    s << '(' << m(0, 0) << ' ' << m(0, 1) << ' ' << m(0, 2) << " ; "
             << m(1, 0) << ' ' << m(1, 1) << ' ' << m(1, 2) << " ; "
             << m(2, 0) << ' ' << m(2, 1) << ' ' << m(2, 2) << ')';
    return s;
}

