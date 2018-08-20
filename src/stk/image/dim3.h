#pragma once

#include <stdint.h>
#include <sstream>

#include <stk/math/int3.h>

#ifdef STK_USE_CUDA
#include <vector_types.h>

#else

// Otherwise defined in vector_types.h for CUDA
struct dim3
{
    dim3(uint32_t vx = 1, uint32_t vy = 1, uint32_t vz = 1) 
        : x(vx), y(vy), z(vz) {}

    uint32_t x;
    uint32_t y;
    uint32_t z;
};

#endif

inline bool operator==(const dim3& l, const dim3& r)
{
    return (l.x == r.x && l.y == r.y && l.z == r.z);
}
inline bool operator!=(const dim3& l, const dim3& r)
{
    return !operator==(l, r);
}

inline std::ostream& operator<<(std::ostream& s, const dim3& v)
{
    s << '(' << v.x << ' ' << v.y << ' ' << v.z << ')';
    return s;
}

namespace stk
{
    // Check whether the given point is inside the given range
    inline bool is_inside(const dim3& dims, const int3& p)
    {
        return (p.x >= 0 && p.x < int(dims.x) && p.y >= 0 && p.y < int(dims.y) && p.z >= 0 && p.z < int(dims.z));
    }
}

