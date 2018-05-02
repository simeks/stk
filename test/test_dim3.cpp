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