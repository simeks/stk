#include "catch.hpp"

#include <stk/common/error.h>
#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>

#include <cfloat>
#include <random>

using namespace stk;

namespace {
    const uint32_t W = 2;
    const uint32_t H = 3;
    const uint32_t D = 4;
}

TEST_CASE("gpu_volume", "[gpu_volume]")
{
    SECTION("constructor")
    {
        GpuVolume vol({W,H,D}, Type_UChar, gpu::Usage_PitchedPointer);
        REQUIRE(vol.size() == dim3{W, H, D});
        REQUIRE(vol.usage() == gpu::Usage_PitchedPointer);
        REQUIRE(vol.valid());
        REQUIRE(vol.pitched_ptr().ptr);

        GpuVolume vol2({W,H,D}, Type_UChar, gpu::Usage_Texture);
        REQUIRE(vol2.size() == dim3{W, H, D});
        REQUIRE(vol2.usage() == gpu::Usage_Texture);
        REQUIRE(vol2.valid());
        REQUIRE(vol2.array_ptr());
    }
    SECTION("allocate")
    {
        GpuVolume vol;
        vol.allocate({W,H,D}, Type_UChar, gpu::Usage_PitchedPointer);
        REQUIRE(vol.size() == dim3{W, H, D});
        REQUIRE(vol.usage() == gpu::Usage_PitchedPointer);
        REQUIRE(vol.valid());
        REQUIRE(vol.pitched_ptr().ptr);

        GpuVolume vol2;
        vol2.allocate({W,H,D}, Type_UChar, gpu::Usage_Texture);
        REQUIRE(vol2.size() == dim3{W, H, D});
        REQUIRE(vol2.usage() == gpu::Usage_Texture);
        REQUIRE(vol2.valid());
        REQUIRE(vol2.array_ptr());
        
    }
    SECTION("release")
    {
        GpuVolume vol;
        vol.allocate({W,H,D}, Type_UChar);
        REQUIRE(vol.size() == dim3{W, H, D});
        REQUIRE(vol.valid());
        REQUIRE(vol.pitched_ptr().ptr);

        vol.release();
        REQUIRE(!vol.valid());
    }
}
TEST_CASE("gpu_volume_ref", "[gpu_volume]")
{
    // Test reference handling

    SECTION("assignment")
    {
        GpuVolume vol({W,H,D}, Type_UChar, gpu::Usage_PitchedPointer);

        // Create a soft copy, will be referencing the same memory
        GpuVolume copy = vol;
        REQUIRE(copy.valid());
        REQUIRE(copy.pitched_ptr().ptr == vol.pitched_ptr().ptr); // Should reference same memory

        // Releasing the old volume should not affect the new copys access
        //  to the data
        vol.release();
        REQUIRE(!vol.valid());
        REQUIRE(copy.valid());
    }
    SECTION("copy_constructor")
    {
        GpuVolume vol({W,H,D}, Type_UChar, gpu::Usage_PitchedPointer);

        // Create a soft copy, will be referencing the same memory
        GpuVolume copy(vol);
        REQUIRE(copy.valid());
        REQUIRE(copy.pitched_ptr().ptr == vol.pitched_ptr().ptr); // Should reference same memory

        // Releasing the old volume should not affect the new copys access
        //  to the data
        vol.release();
        REQUIRE(!vol.valid());
        REQUIRE(copy.valid());
    }
}

TEST_CASE("gpu_volume_types", "[gpu_volume]")
{
    #define TYPE_TEST(T_id) \
        SECTION(#T_id) \
        { \
            GpuVolume vol({W,H,D}, T_id); \
            REQUIRE(vol.valid()); \
        }

    // Types with 3 channels are not supported, and no doubles

    TYPE_TEST(Type_UChar);
    TYPE_TEST(Type_UChar2);
    TYPE_TEST(Type_UChar4);
    TYPE_TEST(Type_Float);
    TYPE_TEST(Type_Float2);
    TYPE_TEST(Type_Float4);

    #undef TYPE_TEST
}

TEST_CASE("gpu_volume_meta_data", "[gpu_volume]")
{
    GpuVolume vol({4,4,4}, Type_Float);

    REQUIRE(vol.origin().x == Approx(0.0f));
    REQUIRE(vol.origin().y == Approx(0.0f));
    REQUIRE(vol.origin().z == Approx(0.0f));

    REQUIRE(vol.spacing().x == Approx(1.0f));
    REQUIRE(vol.spacing().y == Approx(1.0f));
    REQUIRE(vol.spacing().z == Approx(1.0f));

    vol.set_origin({2.0f, 3.0f, 4.0f});
    vol.set_spacing({5.0f, 6.0f, 7.0f});

    REQUIRE(vol.origin().x == Approx(2.0f));
    REQUIRE(vol.origin().y == Approx(3.0f));
    REQUIRE(vol.origin().z == Approx(4.0f));

    REQUIRE(vol.spacing().x == Approx(5.0f));
    REQUIRE(vol.spacing().y == Approx(6.0f));
    REQUIRE(vol.spacing().z == Approx(7.0f));
}

TEST_CASE("gpu_volume_copy_meta", "[gpu_volume]")
{
    GpuVolume a({4,4,4}, Type_Float);

    a.set_origin({2.0f, 3.0f, 4.0f});
    a.set_spacing({5.0f, 6.0f, 7.0f});

    GpuVolume b({2,2,2}, Type_Float);
    b.copy_meta_from(a);

    REQUIRE(b.origin().x == Approx(2.0f));
    REQUIRE(b.origin().y == Approx(3.0f));
    REQUIRE(b.origin().z == Approx(4.0f));

    REQUIRE(b.spacing().x == Approx(5.0f));
    REQUIRE(b.spacing().y == Approx(6.0f));
    REQUIRE(b.spacing().z == Approx(7.0f));
}

TEST_CASE("gpu_volume_upload_download", "[gpu_volume]")
{
    float test_data[W*H*D];
    for (int i = 0; i < int(W*H*D); ++i)
        test_data[i] = float(i);

    Volume vol({W,H,D}, Type_Float, test_data);
    vol.set_origin({2.0f, 3.0f, 4.0f});
    vol.set_spacing({5.0f, 6.0f, 7.0f});
    REQUIRE(vol.valid());

    SECTION("constructor_pitched_pointer")
    {
        GpuVolume gpu_vol(vol, gpu::Usage_PitchedPointer);
        REQUIRE(gpu_vol.valid());
        REQUIRE(gpu_vol.size() == vol.size());
        REQUIRE(gpu_vol.voxel_type() == vol.voxel_type());

        // Should keep the meta data
        REQUIRE(gpu_vol.origin().x == Approx(vol.origin().x));
        REQUIRE(gpu_vol.origin().y == Approx(vol.origin().y));
        REQUIRE(gpu_vol.origin().z == Approx(vol.origin().z));
        REQUIRE(gpu_vol.spacing().x == Approx(vol.spacing().x));
        REQUIRE(gpu_vol.spacing().y == Approx(vol.spacing().y));
        REQUIRE(gpu_vol.spacing().z == Approx(vol.spacing().z));

        Volume vol2 = gpu_vol.download();
        REQUIRE(vol2.valid());
        REQUIRE(vol2.size() == vol.size());
        REQUIRE(vol2.voxel_type() == vol.voxel_type());

        REQUIRE(vol2.origin().x == Approx(vol.origin().x));
        REQUIRE(vol2.origin().y == Approx(vol.origin().y));
        REQUIRE(vol2.origin().z == Approx(vol.origin().z));
        REQUIRE(vol2.spacing().x == Approx(vol.spacing().x));
        REQUIRE(vol2.spacing().y == Approx(vol.spacing().y));
        REQUIRE(vol2.spacing().z == Approx(vol.spacing().z));

        for (int i = 0; i < int(W*H*D); ++i) {
            REQUIRE(static_cast<float*>(vol2.ptr())[i] == Approx(test_data[i]));
        }
    }
    SECTION("constructor_texture")
    {
        GpuVolume gpu_vol(vol, gpu::Usage_Texture);
        REQUIRE(gpu_vol.valid());
        REQUIRE(gpu_vol.size() == vol.size());
        REQUIRE(gpu_vol.voxel_type() == vol.voxel_type());

        // Should keep the meta data
        REQUIRE(gpu_vol.origin().x == Approx(vol.origin().x));
        REQUIRE(gpu_vol.origin().y == Approx(vol.origin().y));
        REQUIRE(gpu_vol.origin().z == Approx(vol.origin().z));
        REQUIRE(gpu_vol.spacing().x == Approx(vol.spacing().x));
        REQUIRE(gpu_vol.spacing().y == Approx(vol.spacing().y));
        REQUIRE(gpu_vol.spacing().z == Approx(vol.spacing().z));

        Volume vol2 = gpu_vol.download();
        REQUIRE(vol2.valid());
        REQUIRE(vol2.size() == vol.size());
        REQUIRE(vol2.voxel_type() == vol.voxel_type());

        REQUIRE(vol2.origin().x == Approx(vol.origin().x));
        REQUIRE(vol2.origin().y == Approx(vol.origin().y));
        REQUIRE(vol2.origin().z == Approx(vol.origin().z));
        REQUIRE(vol2.spacing().x == Approx(vol.spacing().x));
        REQUIRE(vol2.spacing().y == Approx(vol.spacing().y));
        REQUIRE(vol2.spacing().z == Approx(vol.spacing().z));

        for (int i = 0; i < int(W*H*D); ++i) {
            REQUIRE(static_cast<float*>(vol2.ptr())[i] == Approx(test_data[i]));
        }
    }
    SECTION("method_pitched_pointer") // In-place download
    {
        GpuVolume gpu_vol(vol.size(), vol.voxel_type(), gpu::Usage_PitchedPointer);
        gpu_vol.upload(vol);
        REQUIRE(gpu_vol.valid());
        REQUIRE(gpu_vol.size() == vol.size());
        REQUIRE(gpu_vol.voxel_type() == vol.voxel_type());

        // Should keep the meta data
        REQUIRE(gpu_vol.origin().x == Approx(vol.origin().x));
        REQUIRE(gpu_vol.origin().y == Approx(vol.origin().y));
        REQUIRE(gpu_vol.origin().z == Approx(vol.origin().z));
        REQUIRE(gpu_vol.spacing().x == Approx(vol.spacing().x));
        REQUIRE(gpu_vol.spacing().y == Approx(vol.spacing().y));
        REQUIRE(gpu_vol.spacing().z == Approx(vol.spacing().z));

        Volume vol2 = vol.clone(); // For in-place download
        gpu_vol.download(vol2);
        REQUIRE(vol2.valid());
        REQUIRE(vol2.size() == vol.size());
        REQUIRE(vol2.voxel_type() == vol.voxel_type());

        REQUIRE(vol2.origin().x == Approx(vol.origin().x));
        REQUIRE(vol2.origin().y == Approx(vol.origin().y));
        REQUIRE(vol2.origin().z == Approx(vol.origin().z));
        REQUIRE(vol2.spacing().x == Approx(vol.spacing().x));
        REQUIRE(vol2.spacing().y == Approx(vol.spacing().y));
        REQUIRE(vol2.spacing().z == Approx(vol.spacing().z));

        for (int i = 0; i < int(W*H*D); ++i) {
            REQUIRE(static_cast<float*>(vol2.ptr())[i] == Approx(test_data[i]));
        }
    }
    SECTION("method_texture") // In-place download
    {
        GpuVolume gpu_vol(vol.size(), vol.voxel_type(), gpu::Usage_Texture);
        gpu_vol.upload(vol);
        REQUIRE(gpu_vol.valid());
        REQUIRE(gpu_vol.size() == vol.size());
        REQUIRE(gpu_vol.voxel_type() == vol.voxel_type());

        // Should keep the meta data
        REQUIRE(gpu_vol.origin().x == Approx(vol.origin().x));
        REQUIRE(gpu_vol.origin().y == Approx(vol.origin().y));
        REQUIRE(gpu_vol.origin().z == Approx(vol.origin().z));
        REQUIRE(gpu_vol.spacing().x == Approx(vol.spacing().x));
        REQUIRE(gpu_vol.spacing().y == Approx(vol.spacing().y));
        REQUIRE(gpu_vol.spacing().z == Approx(vol.spacing().z));

        Volume vol2 = vol.clone(); // For in-place download
        gpu_vol.download(vol2);
        REQUIRE(vol2.valid());
        REQUIRE(vol2.size() == vol.size());
        REQUIRE(vol2.voxel_type() == vol.voxel_type());

        REQUIRE(vol2.origin().x == Approx(vol.origin().x));
        REQUIRE(vol2.origin().y == Approx(vol.origin().y));
        REQUIRE(vol2.origin().z == Approx(vol.origin().z));
        REQUIRE(vol2.spacing().x == Approx(vol.spacing().x));
        REQUIRE(vol2.spacing().y == Approx(vol.spacing().y));
        REQUIRE(vol2.spacing().z == Approx(vol.spacing().z));

        for (int i = 0; i < int(W*H*D); ++i) {
            REQUIRE(static_cast<float*>(vol2.ptr())[i] == Approx(test_data[i]));
        }
    }
}

TEST_CASE("gpu_volume_clone", "[gpu_volume]")
{
    float test_data[W*H*D];
    for (int i = 0; i < int(W*H*D); ++i)
        test_data[i] = float(i);

    Volume vol({W,H,D}, Type_Float, test_data);
    vol.set_origin({2.0f, 3.0f, 4.0f});
    vol.set_spacing({5.0f, 6.0f, 7.0f});
    REQUIRE(vol.valid());

    GpuVolume gpu_vol(vol, gpu::Usage_PitchedPointer);
    GpuVolume gpu_vol_clone = gpu_vol.clone();

    Volume clone = gpu_vol_clone.download();
    REQUIRE(clone.valid());
    REQUIRE(clone.size() == vol.size());
    REQUIRE(clone.voxel_type() == vol.voxel_type());

    // Meta data should be identical
    REQUIRE(clone.origin().x == Approx(vol.origin().x));
    REQUIRE(clone.origin().y == Approx(vol.origin().y));
    REQUIRE(clone.origin().z == Approx(vol.origin().z));
    REQUIRE(clone.spacing().x == Approx(vol.spacing().x));
    REQUIRE(clone.spacing().y == Approx(vol.spacing().y));
    REQUIRE(clone.spacing().z == Approx(vol.spacing().z));
            
    REQUIRE(clone.ptr() != vol.ptr()); // Should not point to the same memory

    for (int i = 0; i < int(W*H*D); ++i) {
        REQUIRE(static_cast<float*>(clone.ptr())[i] == Approx(test_data[i]));
    }
}

TEST_CASE("gpu_volume_clone_as", "[gpu_volume]")
{
    float test_data[W*H*D];
    for (int i = 0; i < int(W*H*D); ++i)
        test_data[i] = float(i);

    Volume vol({W,H,D}, Type_Float, test_data);
    vol.set_origin({2.0f, 3.0f, 4.0f});
    vol.set_spacing({5.0f, 6.0f, 7.0f});

    SECTION("pitched_pointer_to_texture")
    {
        GpuVolume gpu_vol(vol, gpu::Usage_PitchedPointer);
        GpuVolume gpu_vol_clone = gpu_vol.clone_as(gpu::Usage_Texture);
        REQUIRE(gpu_vol_clone.valid());
        REQUIRE(gpu_vol_clone.usage() == gpu::Usage_Texture);

        Volume clone = gpu_vol_clone.download();
        REQUIRE(clone.valid());
        REQUIRE(clone.size() == vol.size());
        REQUIRE(clone.voxel_type() == vol.voxel_type());

        // Meta data should be identical
        REQUIRE(clone.origin().x == Approx(vol.origin().x));
        REQUIRE(clone.origin().y == Approx(vol.origin().y));
        REQUIRE(clone.origin().z == Approx(vol.origin().z));
        REQUIRE(clone.spacing().x == Approx(vol.spacing().x));
        REQUIRE(clone.spacing().y == Approx(vol.spacing().y));
        REQUIRE(clone.spacing().z == Approx(vol.spacing().z));
                
        REQUIRE(clone.ptr() != vol.ptr()); // Should not point to the same memory

        for (int i = 0; i < int(W*H*D); ++i) {
            REQUIRE(static_cast<float*>(clone.ptr())[i] == Approx(test_data[i]));
        }
    }
    SECTION("texture_to_pitched_pointer")
    {
        GpuVolume gpu_vol(vol, gpu::Usage_Texture);
        GpuVolume gpu_vol_clone = gpu_vol.clone_as(gpu::Usage_PitchedPointer);
        REQUIRE(gpu_vol_clone.valid());
        REQUIRE(gpu_vol_clone.usage() == gpu::Usage_PitchedPointer);

        Volume clone = gpu_vol_clone.download();
        REQUIRE(clone.valid());
        REQUIRE(clone.size() == vol.size());
        REQUIRE(clone.voxel_type() == vol.voxel_type());

        // Meta data should be identical
        REQUIRE(clone.origin().x == Approx(vol.origin().x));
        REQUIRE(clone.origin().y == Approx(vol.origin().y));
        REQUIRE(clone.origin().z == Approx(vol.origin().z));
        REQUIRE(clone.spacing().x == Approx(vol.spacing().x));
        REQUIRE(clone.spacing().y == Approx(vol.spacing().y));
        REQUIRE(clone.spacing().z == Approx(vol.spacing().z));
                
        REQUIRE(clone.ptr() != vol.ptr()); // Should not point to the same memory

        for (int i = 0; i < int(W*H*D); ++i) {
            REQUIRE(static_cast<float*>(clone.ptr())[i] == Approx(test_data[i]));
        }
    }
}

TEST_CASE("gpu_volume_copy_from", "[gpu_volume]")
{
    float test_data[W*H*D];
    for (int i = 0; i < int(W*H*D); ++i)
        test_data[i] = float(i);

    Volume vol({W,H,D}, Type_Float, test_data);
    vol.set_origin({2.0f, 3.0f, 4.0f});
    vol.set_spacing({5.0f, 6.0f, 7.0f});
    REQUIRE(vol.valid());

    GpuVolume gpu_vol(vol, gpu::Usage_PitchedPointer);
    GpuVolume gpu_vol_clone(vol.size(), vol.voxel_type(), gpu::Usage_PitchedPointer);
    
    gpu_vol_clone.copy_from(gpu_vol);
    REQUIRE(gpu_vol_clone.valid());
    REQUIRE(gpu_vol_clone.usage() == gpu::Usage_PitchedPointer);

    Volume clone = gpu_vol_clone.download();
    REQUIRE(clone.valid());
    REQUIRE(clone.size() == vol.size());
    REQUIRE(clone.voxel_type() == vol.voxel_type());

    // Meta data should be identical
    REQUIRE(clone.origin().x == Approx(vol.origin().x));
    REQUIRE(clone.origin().y == Approx(vol.origin().y));
    REQUIRE(clone.origin().z == Approx(vol.origin().z));
    REQUIRE(clone.spacing().x == Approx(vol.spacing().x));
    REQUIRE(clone.spacing().y == Approx(vol.spacing().y));
    REQUIRE(clone.spacing().z == Approx(vol.spacing().z));
            
    REQUIRE(clone.ptr() != vol.ptr()); // Should not point to the same memory

    for (int i = 0; i < int(W*H*D); ++i) {
        REQUIRE(static_cast<float*>(clone.ptr())[i] == Approx(test_data[i]));
    }
}

TEST_CASE("gpu_volume_min_max", "[gpu_volume]")
{
    dim3 sizes[] = {
        {8, 8, 8},
        {128, 128, 128},
        {128, 32, 8},
        {128, 32, 1},
        {128, 1, 1},
        {1, 128, 1},
        {1, 1, 128},
        {1, 1, 1},
    };

    for (int s = 0; s < 8; ++s) {
        dim3 dim = sizes[s];

        float* test_data = new float[dim.x*dim.y*dim.z];
    
        std::random_device rd;
        std::mt19937 gen(4321);
        std::uniform_int_distribution<> dis(0, 10000000);

        float true_min = FLT_MAX;
        float true_max = -FLT_MAX;
        for (uint32_t i = 0; i < dim.x*dim.y*dim.z; ++i) {
            test_data[i] = (float)dis(gen);

            true_min = std::min(true_min, test_data[i]);
            true_max = std::max(true_max, test_data[i]);
        }

        Volume vol(dim, Type_Float, test_data);
        REQUIRE(vol.valid());

        GpuVolume gpu_vol(vol, gpu::Usage_PitchedPointer);
        REQUIRE(gpu_vol.valid());
        
        float min, max;
        find_min_max(gpu_vol, min, max);

        REQUIRE(min == Approx(true_min));
        REQUIRE(max == Approx(true_max));

        delete [] test_data;
    }
}

