#include "catch.hpp"

#include <stk/filters/gpu/normalize.h>
#include <stk/image/gpu_volume.h>

using namespace stk;

TEST_CASE("gpu_normalize", "[volume]")
{
    SECTION("texture")
    {
        float data[] = {
            4, 2, 
            1, 0,
            
            4, 2, 
            1, 0
        };
        VolumeFloat vol({2,2,2}, data);
        GpuVolume gpu_vol(vol, gpu::Usage_Texture);
        
        GpuVolume gpu_vol_norm = gpu::normalize(gpu_vol, 0, 1);
        VolumeFloat vol_norm = gpu_vol_norm.download();

        CHECK(vol_norm(0,0,0) == Approx(1.0f));
        CHECK(vol_norm(1,0,0) == Approx(0.5f));
        CHECK(vol_norm(0,1,0) == Approx(0.25f));
        CHECK(vol_norm(1,1,0) == Approx(0.0f));

        CHECK(vol_norm(0,0,1) == Approx(1.0f));
        CHECK(vol_norm(1,0,1) == Approx(0.5f));
        CHECK(vol_norm(0,1,1) == Approx(0.25f));
        CHECK(vol_norm(1,1,1) == Approx(0.0f));
    }
    SECTION("pitched_pointer")
    {
        float data[] = {
            4, 2, 
            1, 0,
            
            4, 2, 
            1, 0
        };
        VolumeFloat vol({2,2,2}, data);
        GpuVolume gpu_vol(vol, gpu::Usage_PitchedPointer);
        
        GpuVolume gpu_vol_norm = gpu::normalize(gpu_vol, 0, 1);
        VolumeFloat vol_norm = gpu_vol_norm.download();

        CHECK(vol_norm(0,0,0) == Approx(1.0f));
        CHECK(vol_norm(1,0,0) == Approx(0.5f));
        CHECK(vol_norm(0,1,0) == Approx(0.25f));
        CHECK(vol_norm(1,1,0) == Approx(0.0f));

        CHECK(vol_norm(0,0,1) == Approx(1.0f));
        CHECK(vol_norm(1,0,1) == Approx(0.5f));
        CHECK(vol_norm(0,1,1) == Approx(0.25f));
        CHECK(vol_norm(1,1,1) == Approx(0.0f));
    }
    SECTION("in_place_pitched_pointer")
    {
        float data[] = {
            4, 2, 
            1, 0,
            
            4, 2, 
            1, 0
        };
        VolumeFloat vol({2,2,2}, data);
        GpuVolume gpu_vol(vol, gpu::Usage_Texture);
        
        gpu::normalize(gpu_vol, 0, 1, &gpu_vol);
        VolumeFloat vol_norm = gpu_vol.download();

        CHECK(vol_norm(0,0,0) == Approx(1.0f));
        CHECK(vol_norm(1,0,0) == Approx(0.5f));
        CHECK(vol_norm(0,1,0) == Approx(0.25f));
        CHECK(vol_norm(1,1,0) == Approx(0.0f));

        CHECK(vol_norm(0,0,1) == Approx(1.0f));
        CHECK(vol_norm(1,0,1) == Approx(0.5f));
        CHECK(vol_norm(0,1,1) == Approx(0.25f));
        CHECK(vol_norm(1,1,1) == Approx(0.0f));
    }
    SECTION("in_place_pitched_pointer")
    {
        float data[] = {
            4, 2, 
            1, 0,
            
            4, 2, 
            1, 0
        };
        VolumeFloat vol({2,2,2}, data);
        GpuVolume gpu_vol(vol, gpu::Usage_PitchedPointer);
        
        gpu::normalize(gpu_vol, 0, 1, &gpu_vol);
        VolumeFloat vol_norm = gpu_vol.download();

        CHECK(vol_norm(0,0,0) == Approx(1.0f));
        CHECK(vol_norm(1,0,0) == Approx(0.5f));
        CHECK(vol_norm(0,1,0) == Approx(0.25f));
        CHECK(vol_norm(1,1,0) == Approx(0.0f));

        CHECK(vol_norm(0,0,1) == Approx(1.0f));
        CHECK(vol_norm(1,0,1) == Approx(0.5f));
        CHECK(vol_norm(0,1,1) == Approx(0.25f));
        CHECK(vol_norm(1,1,1) == Approx(0.0f));
    }
}

