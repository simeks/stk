#include "catch.hpp"

#include <stk/image/dim3.h>

TEST_CASE("dim3 op", "[volume]")
{
    dim3 dims{10, 20, 30};

    REQUIRE(dims == dim3{10, 20, 30});
    REQUIRE(dims != dim3{99, 20, 30});
    REQUIRE(dims != dim3{10, 99, 30});
    REQUIRE(dims != dim3{10, 20, 99});
}
TEST_CASE("dim3_is_inside", "[volume]")
{
    dim3 dims{10, 10, 10};

    REQUIRE(stk::is_inside(dims, int3{0,0,0}));
    REQUIRE(stk::is_inside(dims, int3{5,5,5}));
    
    REQUIRE(!stk::is_inside(dims, int3{10,10,10}));
    REQUIRE(!stk::is_inside(dims, int3{10,5,5}));
    REQUIRE(!stk::is_inside(dims, int3{5,10,5}));
    REQUIRE(!stk::is_inside(dims, int3{5,5,10}));

    REQUIRE(!stk::is_inside(dims, int3{-5,5,5}));
    REQUIRE(!stk::is_inside(dims, int3{5,-5,5}));
    REQUIRE(!stk::is_inside(dims, int3{5,5,-5}));
}
TEST_CASE("dim3_to_string", "[volume]")
{
    std::ostringstream s;
    s << dim3{10, 20, 30};
    REQUIRE(s.str() == "(10 20 30)");
}
TEST_CASE("dim3_iterator", "[volume]")
{
    dim3 d{3, 5, 7};
    
    int3 r_expected{0,0,0};
    for (int z = 0; z < (int)d.z; ++z) {
    for (int y = 0; y < (int)d.y; ++y) {
    for (int x = 0; x < (int)d.x; ++x) {
        r_expected.x += x;
        r_expected.y += y;
        r_expected.z += z;
    }}}

    int3 r{0,0,0};

    for (int3 i : d) {
        r.x += i.x;
        r.y += i.y;
        r.z += i.z;
    }
    REQUIRE(r.x == r_expected.x);
    REQUIRE(r.y == r_expected.y);
    REQUIRE(r.z == r_expected.z);
}
#ifndef _WIN32
// Windows only supports OpenMP 2, which means no support for iterators.
TEST_CASE("dim3_iterator_omp", "[volume]")
{
    dim3 d{3, 5, 7};
    
    int3 r_expected{0,0,0};
    for (int z = 0; z < (int)d.z; ++z) {
    for (int y = 0; y < (int)d.y; ++y) {
    for (int x = 0; x < (int)d.x; ++x) {
        r_expected.x += x;
        r_expected.y += y;
        r_expected.z += z;
    }}}

    int rx = 0, ry = 0, rz = 0;

    #pragma omp parallel for reduction(+: rx, ry, rz)
    for (Dim3Iterator it = begin(d); it < end(d); ++it) {
        rx += (*it).x;
        ry += (*it).y;
        rz += (*it).z;
    }
    REQUIRE(rx == r_expected.x);
    REQUIRE(ry == r_expected.y);
    REQUIRE(rz == r_expected.z);
}
#endif

