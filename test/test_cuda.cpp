#include "catch.hpp"

#include <stk/cuda/cuda.h>

using namespace stk;

TEST_CASE("cuda", "[cuda]")
{
    REQUIRE_NOTHROW(cuda::init());

    int device_count = 0;
    REQUIRE_NOTHROW(device_count = cuda::device_count());
    REQUIRE(device_count > 0);

    for (int i = 0; i < device_count; ++i) {
        REQUIRE_NOTHROW(cuda::set_device(i));
        REQUIRE(cuda::device() == i);
    }
}

