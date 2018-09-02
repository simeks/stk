#pragma once

#include "types.h"

// Element-wise multiplication
inline int3 operator*(const int3& l, const int3& r)
{
    return int3{l.x*r.x, l.y*r.y, l.z*r.z};
}
inline int3 operator/(const int3& l, const int3& r)
{
    return int3{l.x/r.x, l.y/r.y, l.z/r.z};
}
inline int3 operator+(const int3& l, const int3& r)
{
    return int3{l.x+r.x, l.y+r.y, l.z+r.z};
}
inline int3 operator-(const int3& l, const int3& r)
{
    return int3{l.x-r.x, l.y-r.y, l.z-r.z};
}
inline int3 operator*(const int l, const int3& r)
{
    return int3{l*r.x, l*r.y, l*r.z};
}
inline int3 operator*(const int3& l, const int r)
{
    return int3{r*l.x, r*l.y, r*l.z};
}
inline int3 operator/(const int3& l, const int r)
{
    return int3{l.x/r, l.y/r, l.z/r};
}
