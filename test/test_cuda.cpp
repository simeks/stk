#include "catch.hpp"

#include <stk/cuda/cuda.h>
#include <stk/image/volume.h>

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
TEST_CASE("cuda_pinned_memory", "[cuda]")
{
    REQUIRE_NOTHROW(cuda::init());

    dim3 dims[] = {
        {32,32,32},
        {64,64,64},
        {128,128,128},
        {256,256,256},
        {512,512,512}
    };

    for (int i = 0; i < 5; ++i) {
        Volume vol;
        REQUIRE_NOTHROW(vol.allocate(dims[i], Type_Float, Usage_Pinned));
        REQUIRE(vol.ptr());
        REQUIRE_NOTHROW(vol.release());

        REQUIRE_NOTHROW(vol.allocate(dims[i], Type_Float, Usage_Mapped));
        REQUIRE(vol.ptr());
        REQUIRE_NOTHROW(vol.release());

        REQUIRE_NOTHROW(vol.allocate(dims[i], Type_Float, Usage_WriteCombined));
        REQUIRE(vol.ptr());
        REQUIRE_NOTHROW(vol.release());
    }
}



