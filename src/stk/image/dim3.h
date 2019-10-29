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

/** Iterator to simplify iterating through a region.
 * 
 * Given a dim3, it holds a int3 pointing to the current position. Advancing
 * the iterator pushes the pointer to the next position until reaching 
 * {0, 0, dim3.z}, which is considered the end.
 * 
 * Example:
 * The preferred way is to use C++11 range-based for loops:
 *
 *  stk::Volume vol;
 *  for (int3 p : vol.size()) {
 *      vol(p) = 0:
 *  }
 * 
 * To use the iterators with OpenMP you'll have to manually set up the loop, and also
 * remember to use 'it < end()', rather than the typical 'it != end()'.
 *
 * Note: Windows only supports OpenMP 2.0, meaning there's no support for iterators.
 *
 *  stk::Volume vol;
 *  #pragma omp parallel for
 *  for (auto it = begin(vol.size()); it < end(vol.size()); ++it) {
 *      vol(*it) = 0;
 *  }
 * 
 * */
struct Dim3Iterator
{
    dim3 _dim;
    int3 _p;

    Dim3Iterator(dim3 dim, int3 p) : _dim(dim), _p(p) {}

    inline const int3& operator*()
    {
        return _p;
    }
    inline const int3* operator->()
    {
        return &_p;
    }
    inline void operator++()
    {
        _p.x += 1;
        if (_p.x == (int)_dim.x) {
            _p.x = 0;
            _p.y += 1;
            if (_p.y == (int)_dim.y) {
                _p.y = 0;
                _p.z += 1;
            }
        }
    }
    inline bool operator!=(const Dim3Iterator& other)
    {
        return index() != other.index();
    }
    // Required by OpenMP
    inline bool operator<(const Dim3Iterator& other)
    {
        return index() < other.index();
    }
    // Required by OpenMP
    inline void operator+=(size_t n)
    {
        for (size_t i = 0; i < n; ++i) ++(*this);
    }
    inline size_t index() const
    {
        return _p.x + _p.y * _dim.x + _p.z * _dim.x * _dim.y;
    }
};

// Required by OpenMP
inline size_t operator-(const Dim3Iterator& a, const Dim3Iterator& b)
{
    return a.index() - b.index();
}

// Retrieves the begin iterator for Dim3Iterator.
inline Dim3Iterator begin(const dim3& d)
{
    return Dim3Iterator(d, int3{0, 0, 0});
}

// Retrieves the end iterator for Dim3Iterator.
//  p{0,0,d.z} is defined as the end.
inline Dim3Iterator end(const dim3& d)
{
    return Dim3Iterator(d, int3{0, 0, int(d.z)});
}

namespace stk
{
    // Check whether the given point is inside the given range
    inline bool is_inside(const dim3& dims, const int3& p)
    {
        return (p.x >= 0 && p.x < int(dims.x) && 
                p.y >= 0 && p.y < int(dims.y) && 
                p.z >= 0 && p.z < int(dims.z));
    }
}

