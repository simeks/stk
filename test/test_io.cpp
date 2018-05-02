#include "catch.hpp"

#include "test_util.h"

#include <stk/common/error.h>
#include <stk/io/io.h>
#include <stk/image/volume.h>

using namespace stk;

namespace {
    const int W = 2;
    const int H = 3;
    const int D = 4;
}

TEST_CASE("io", "[io] [volume]")
{
    Volume vol1 = read_volume("dont_exists.vtk");
    REQUIRE(!vol1.valid());

    Volume vol2 = read_volume("dont_exists.not_supported");
    REQUIRE(!vol2.valid());
}

#define IO_TEST_TYPE(ext, T) \
    SECTION(#T) { \
        T test_data[W*H*D]; \
        TestDataGenerator<T>::run(test_data, W, H, D); \
        VolumeHelper<T> vol({W,H,D}, test_data); \
        vol.set_origin({2,3,4}); \
        vol.set_spacing({5,6,7}); \
        write_volume("test_file_" #T "." #ext, vol); \
        VolumeHelper<T> read_vol = read_volume("test_file_" #T "." #ext); \
        REQUIRE(read_vol.valid()); \
        REQUIRE(compare_volumes(read_vol, vol)); \
        CHECK(read_vol.origin().x == Approx(vol.origin().x)); \
        CHECK(read_vol.origin().y == Approx(vol.origin().y)); \
        CHECK(read_vol.origin().z == Approx(vol.origin().z)); \
        CHECK(read_vol.spacing().x == Approx(vol.spacing().x)); \
        CHECK(read_vol.spacing().y == Approx(vol.spacing().y)); \
        CHECK(read_vol.spacing().z == Approx(vol.spacing().z)); \
    }

#define IO_TEST_EXTENSION(ext) \
    TEST_CASE("io_"#ext, "[io] [volume]") { \
        IO_TEST_TYPE(ext, float); \
        IO_TEST_TYPE(ext, float2); \
        IO_TEST_TYPE(ext, float3); \
        IO_TEST_TYPE(ext, float4); \
        IO_TEST_TYPE(ext, double); \
        IO_TEST_TYPE(ext, double2); \
        IO_TEST_TYPE(ext, double3); \
        IO_TEST_TYPE(ext, double4); \
        IO_TEST_TYPE(ext, uint8_t); \
        IO_TEST_TYPE(ext, uchar2); \
        IO_TEST_TYPE(ext, uchar3); \
        IO_TEST_TYPE(ext, uchar4); \
    }

IO_TEST_EXTENSION(vtk);
IO_TEST_EXTENSION(nii);
IO_TEST_EXTENSION(nii.gz);
