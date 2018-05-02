#pragma once

#include "stk/math/types.h"
#include "stk/image/volume.h"

// Misc utils for testing

template<typename T> bool test_equal(T a, T b) { return a == b; }
template<> bool test_equal(float a, float b) 
{ 
    return fabs(a-b) < 1e-5f;
}
template<> bool test_equal(float2 a, float2 b)
{ 
    return test_equal(a.x, b.x) && test_equal(a.y, b.y);
}
template<> bool test_equal(float3 a, float3 b)
{ 
    return test_equal(a.x, b.x) && test_equal(a.y, b.y)
        && test_equal(a.z, b.z);
}
template<> bool test_equal(float4 a, float4 b)
{ 
    return test_equal(a.x, b.x) && test_equal(a.y, b.y) 
        && test_equal(a.z, b.z) && test_equal(a.w, b.w);
}

template<> bool test_equal(double a, double b) 
{ 
    return fabs(a-b) < 1e-5f;
}
template<> bool test_equal(double2 a, double2 b)
{ 
    return test_equal(a.x, b.x) && test_equal(a.y, b.y);
}
template<> bool test_equal(double3  a, double3 b)
{ 
    return test_equal(a.x, b.x) && test_equal(a.y, b.y)
        && test_equal(a.z, b.z);
}
template<> bool test_equal(double4 a, double4 b)
{ 
    return test_equal(a.x, b.x) && test_equal(a.y, b.y) 
        && test_equal(a.z, b.z) && test_equal(a.w, b.w);
}

// Returns true if given volumes are equal
template<typename T>
bool compare_volumes(const stk::VolumeHelper<T>& a, const stk::VolumeHelper<T>& b)
{
    for (int z = 0; z < D; ++z) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (!test_equal(a(x,y,z), b(x,y,z)))
                    return false;
            }
        }
    }
    return true;
}

template<typename T, typename BT=type_id<T>::Base, int NC=type_id<T>::num_comp>
struct TestDataGenerator
{
    static void run(T* out, int w, int h, int d)
    {
        for (int i = 0; i < w*h*d; ++i) out[i] = T(i);
    }
};
template<typename T, typename BT>
struct TestDataGenerator<T, BT, 2>
{
    static void run(T* out, int w, int h, int d)
    {
        for (int i = 0; i < w*h*d; ++i) out[i] = {BT(i), BT(i+1)};
    }
};
template<typename T, typename BT>
struct TestDataGenerator<T, BT, 3>
{
    static void run(T* out, int w, int h, int d)
    {
        for (int i = 0; i < w*h*d; ++i) out[i] = {BT(i), BT(i+1), BT(i+2)};
    }
};
template<typename T, typename BT>
struct TestDataGenerator<T, BT, 4>
{
    static void run(T* out, int w, int h, int d)
    {
        for (int i = 0; i < w*h*d; ++i) out[i] = {BT(i), BT(i+1), BT(i+2), BT(i+3)};
    }
};
