#include "catch.hpp"

#include <stk/filters/normalize.h>
#include <stk/image/volume.h>

using namespace stk;

TEST_CASE("normalize", "[volume]")
{
    SECTION("uint8_t")
    {
        uint8_t data[] = {
            100, 50, 
            25, 0,

            100, 50, 
            25, 0
        };
        VolumeUChar vol({2,2,2}, data);
        VolumeUChar vol_norm = normalize<uint8_t>(vol, 0, 255);

        CHECK((int)vol_norm(0,0,0) == 255);
        CHECK((int)vol_norm(1,0,0) == 127);
        CHECK((int)vol_norm(0,1,0) == 63);
        CHECK((int)vol_norm(1,1,0) == 0);

        CHECK((int)vol_norm(0,0,1) == 255);
        CHECK((int)vol_norm(1,0,1) == 127);
        CHECK((int)vol_norm(0,1,1) == 63);
        CHECK((int)vol_norm(1,1,1) == 0);
    }
    SECTION("uint32_t")
    {
        uint32_t data[] = {
            1000, 500, 
            250, 0,

            1000, 500, 
            250, 0
        };
        VolumeUInt vol({2,2,2}, data);
        VolumeUInt vol_norm = normalize<uint32_t>(vol, 0, 255);

        CHECK(vol_norm(0,0,0) == 255);
        CHECK(vol_norm(1,0,0) == 127);
        CHECK(vol_norm(0,1,0) == 63);
        CHECK(vol_norm(1,1,0) == 0);

        CHECK(vol_norm(0,0,1) == 255);
        CHECK(vol_norm(1,0,1) == 127);
        CHECK(vol_norm(0,1,1) == 63);
        CHECK(vol_norm(1,1,1) == 0);
    }
    SECTION("float")
    {
        float data[] = {
            4, 2, 
            1, 0,
            
            4, 2, 
            1, 0
        };
        VolumeFloat vol({2,2,2}, data);
        VolumeFloat vol_norm = normalize<float>(vol, 0, 1);

        CHECK(vol_norm(0,0,0) == Approx(1.0f));
        CHECK(vol_norm(1,0,0) == Approx(0.5f));
        CHECK(vol_norm(0,1,0) == Approx(0.25f));
        CHECK(vol_norm(1,1,0) == Approx(0.0f));

        CHECK(vol_norm(0,0,1) == Approx(1.0f));
        CHECK(vol_norm(1,0,1) == Approx(0.5f));
        CHECK(vol_norm(0,1,1) == Approx(0.25f));
        CHECK(vol_norm(1,1,1) == Approx(0.0f));
    }
    SECTION("in_place")
    {
        float data[] = {
            4, 2, 
            1, 0,
            
            4, 2, 
            1, 0
        };
        VolumeFloat vol({4,1,1}, data);
        VolumeFloat vol_norm = normalize<float>(vol, 0, 1, &vol);
        CHECK(vol_norm.ptr() == vol.ptr());

        CHECK(vol_norm(0,0,0) == Approx(1.0f));
        CHECK(vol_norm(1,0,0) == Approx(0.5f));
        CHECK(vol_norm(2,0,0) == Approx(0.25f));
        CHECK(vol_norm(3,0,0) == Approx(0.0f));

        CHECK(vol(0,0,0) == Approx(1.0f));
        CHECK(vol(1,0,0) == Approx(0.5f));
        CHECK(vol(2,0,0) == Approx(0.25f));
        CHECK(vol(3,0,0) == Approx(0.0f));
    }
}

