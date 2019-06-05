#include "catch.hpp"

#include <stk/common/error.h>
#include <stk/cuda/stream.h>
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

TEST_CASE("gpu_volume_copy_from_async", "[gpu_volume]")
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

    cuda::Stream stream;
    gpu_vol_clone.copy_from(gpu_vol, stream);
    stream.synchronize();
    
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
TEST_CASE("gpu_volume_upload_download_async", "[gpu_volume]")
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
        cuda::Stream stream;

        GpuVolume gpu_vol(vol, stream, gpu::Usage_PitchedPointer);
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

        Volume vol2 = gpu_vol.download(stream);

        REQUIRE(vol2.valid());
        REQUIRE(vol2.size() == vol.size());
        REQUIRE(vol2.voxel_type() == vol.voxel_type());

        REQUIRE(vol2.origin().x == Approx(vol.origin().x));
        REQUIRE(vol2.origin().y == Approx(vol.origin().y));
        REQUIRE(vol2.origin().z == Approx(vol.origin().z));
        REQUIRE(vol2.spacing().x == Approx(vol.spacing().x));
        REQUIRE(vol2.spacing().y == Approx(vol.spacing().y));
        REQUIRE(vol2.spacing().z == Approx(vol.spacing().z));

        stream.synchronize(); // Synchronize to make sure data is ready

        for (int i = 0; i < int(W*H*D); ++i) {
            REQUIRE(static_cast<float*>(vol2.ptr())[i] == Approx(test_data[i]));
        }
    }
    SECTION("constructor_texture")
    {
        cuda::Stream stream;

        GpuVolume gpu_vol(vol, stream, gpu::Usage_Texture);
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

        Volume vol2 = gpu_vol.download(stream);
        REQUIRE(vol2.valid());
        REQUIRE(vol2.size() == vol.size());
        REQUIRE(vol2.voxel_type() == vol.voxel_type());

        REQUIRE(vol2.origin().x == Approx(vol.origin().x));
        REQUIRE(vol2.origin().y == Approx(vol.origin().y));
        REQUIRE(vol2.origin().z == Approx(vol.origin().z));
        REQUIRE(vol2.spacing().x == Approx(vol.spacing().x));
        REQUIRE(vol2.spacing().y == Approx(vol.spacing().y));
        REQUIRE(vol2.spacing().z == Approx(vol.spacing().z));

        stream.synchronize(); // Synchronize to make sure data is ready

        for (int i = 0; i < int(W*H*D); ++i) {
            REQUIRE(static_cast<float*>(vol2.ptr())[i] == Approx(test_data[i]));
        }
    }
    SECTION("method_pitched_pointer") // In-place download
    {
        cuda::Stream stream;

        GpuVolume gpu_vol(vol.size(), vol.voxel_type(), gpu::Usage_PitchedPointer);
        gpu_vol.upload(vol, stream);
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
        gpu_vol.download(vol2, stream);

        REQUIRE(vol2.valid());
        REQUIRE(vol2.size() == vol.size());
        REQUIRE(vol2.voxel_type() == vol.voxel_type());

        REQUIRE(vol2.origin().x == Approx(vol.origin().x));
        REQUIRE(vol2.origin().y == Approx(vol.origin().y));
        REQUIRE(vol2.origin().z == Approx(vol.origin().z));
        REQUIRE(vol2.spacing().x == Approx(vol.spacing().x));
        REQUIRE(vol2.spacing().y == Approx(vol.spacing().y));
        REQUIRE(vol2.spacing().z == Approx(vol.spacing().z));

        stream.synchronize(); // Synchronize to make sure data is ready

        for (int i = 0; i < int(W*H*D); ++i) {
            REQUIRE(static_cast<float*>(vol2.ptr())[i] == Approx(test_data[i]));
        }
    }
    SECTION("method_texture") // In-place download
    {
        cuda::Stream stream;

        GpuVolume gpu_vol(vol.size(), vol.voxel_type(), gpu::Usage_Texture);
        gpu_vol.upload(vol, stream);
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
        gpu_vol.download(vol2, stream);
        REQUIRE(vol2.valid());
        REQUIRE(vol2.size() == vol.size());
        REQUIRE(vol2.voxel_type() == vol.voxel_type());

        REQUIRE(vol2.origin().x == Approx(vol.origin().x));
        REQUIRE(vol2.origin().y == Approx(vol.origin().y));
        REQUIRE(vol2.origin().z == Approx(vol.origin().z));
        REQUIRE(vol2.spacing().x == Approx(vol.spacing().x));
        REQUIRE(vol2.spacing().y == Approx(vol.spacing().y));
        REQUIRE(vol2.spacing().z == Approx(vol.spacing().z));

        stream.synchronize(); // Synchronize to make sure data is ready

        for (int i = 0; i < int(W*H*D); ++i) {
            REQUIRE(static_cast<float*>(vol2.ptr())[i] == Approx(test_data[i]));
        }
    }
}
TEST_CASE("gpu_volume_clone_async", "[gpu_volume]")
{
    cuda::Stream stream;
    
    float test_data[W*H*D];
    for (int i = 0; i < int(W*H*D); ++i)
        test_data[i] = float(i);

    Volume vol({W,H,D}, Type_Float, test_data);
    vol.set_origin({2.0f, 3.0f, 4.0f});
    vol.set_spacing({5.0f, 6.0f, 7.0f});
    REQUIRE(vol.valid());

    GpuVolume gpu_vol(vol, gpu::Usage_PitchedPointer);
    GpuVolume gpu_vol_clone = gpu_vol.clone(stream);

    stream.synchronize();

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
TEST_CASE("gpu_volume_clone_as_async", "[gpu_volume]")
{
    float test_data[W*H*D];
    for (int i = 0; i < int(W*H*D); ++i)
        test_data[i] = float(i);

    Volume vol({W,H,D}, Type_Float, test_data);
    vol.set_origin({2.0f, 3.0f, 4.0f});
    vol.set_spacing({5.0f, 6.0f, 7.0f});

    SECTION("pitched_pointer_to_texture")
    {
        cuda::Stream stream;

        GpuVolume gpu_vol(vol, gpu::Usage_PitchedPointer);
        GpuVolume gpu_vol_clone = gpu_vol.clone_as(gpu::Usage_Texture, stream);
        REQUIRE(gpu_vol_clone.valid());
        REQUIRE(gpu_vol_clone.usage() == gpu::Usage_Texture);

        Volume clone = gpu_vol_clone.download(stream);
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

        stream.synchronize(); // Synchronize to make sure data is ready

        for (int i = 0; i < int(W*H*D); ++i) {
            REQUIRE(static_cast<float*>(clone.ptr())[i] == Approx(test_data[i]));
        }
    }
    SECTION("texture_to_pitched_pointer")
    {
        cuda::Stream stream;

        GpuVolume gpu_vol(vol, gpu::Usage_Texture);
        GpuVolume gpu_vol_clone = gpu_vol.clone_as(gpu::Usage_PitchedPointer, stream);
        REQUIRE(gpu_vol_clone.valid());
        REQUIRE(gpu_vol_clone.usage() == gpu::Usage_PitchedPointer);

        Volume clone = gpu_vol_clone.download(stream);
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

        stream.synchronize(); // Synchronize to make sure data is ready

        for (int i = 0; i < int(W*H*D); ++i) {
            REQUIRE(static_cast<float*>(clone.ptr())[i] == Approx(test_data[i]));
        }
    }
}
TEST_CASE("gpu_volume_region", "[volume]")
{
    int val[] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16,

        17, 18, 19, 20,
        21, 22, 23, 24,
        25, 26, 27, 28,
        29, 30, 31, 32,

        33, 34, 35, 36,
        37, 38, 39, 40,
        41, 42, 43, 44,
        45, 46, 47, 48,

        49, 50, 51, 52,
        53, 54, 55, 56,
        57, 58, 59, 60,
        61, 62, 63, 64
    };
    int sub_val[] = {
        1, 2,
        3, 4,

        5, 6,
        7, 8
    };

    
    SECTION("constructor") {
        VolumeInt vol({4, 4, 4}, val);
        GpuVolume gpu_vol(vol);
        gpu_vol.set_origin(float3{10.0f, 20.0f, 30.0f});
        gpu_vol.set_spacing(float3{1.0f, 2.0f, 3.0f});
        GpuVolume gpu_sub = gpu_vol({1,4}, {1,4}, {1,4});
        REQUIRE(gpu_sub.size().x == 3);
        REQUIRE(gpu_sub.size().y == 3);
        REQUIRE(gpu_sub.size().z == 3);

        REQUIRE(gpu_sub.origin().x == Approx(11.0f));
        REQUIRE(gpu_sub.origin().y == Approx(22.0f));
        REQUIRE(gpu_sub.origin().z == Approx(33.0f));

        REQUIRE(gpu_sub.spacing().x == Approx(1.0f));
        REQUIRE(gpu_sub.spacing().y == Approx(2.0f));
        REQUIRE(gpu_sub.spacing().z == Approx(3.0f));

        {
            VolumeInt sub = gpu_sub.download();
            REQUIRE(sub(0,0,0) == 22);
            REQUIRE(sub(1,0,0) == 23);
            REQUIRE(sub(0,1,0) == 26);
            REQUIRE(sub(1,1,0) == 27);
            REQUIRE(sub(0,0,1) == 38);
            REQUIRE(sub(1,0,1) == 39);
            REQUIRE(sub(0,1,1) == 42);
            REQUIRE(sub(1,1,1) == 43);
        }

        GpuVolume gpu_sub2(gpu_sub, {1,2}, {1,2}, {0,2});
        REQUIRE(gpu_sub2.size().x == 1);
        REQUIRE(gpu_sub2.size().y == 1);
        REQUIRE(gpu_sub2.size().z == 2);

        REQUIRE(gpu_sub2.origin().x == Approx(12.0f));
        REQUIRE(gpu_sub2.origin().y == Approx(24.0f));
        REQUIRE(gpu_sub2.origin().z == Approx(33.0f));

        {
            VolumeInt sub = gpu_sub2.download();
            REQUIRE(sub(0,0,0) == 27);
            REQUIRE(sub(0,0,1) == 43);
        }

        GpuVolume gpu_sub3(gpu_vol, {0,3}, {0,3}, {0,3});
        REQUIRE(gpu_sub3.size().x == 3);
        REQUIRE(gpu_sub3.size().y == 3);
        REQUIRE(gpu_sub3.size().z == 3);

        REQUIRE(gpu_sub3.origin().x == Approx(10.0f));
        REQUIRE(gpu_sub3.origin().y == Approx(20.0f));
        REQUIRE(gpu_sub3.origin().z == Approx(30.0f));

        {
            VolumeInt sub = gpu_sub3.download();
            REQUIRE(sub(0,0,0) == 1);
            REQUIRE(sub(1,0,0) == 2);
            REQUIRE(sub(2,0,0) == 3);
            
            REQUIRE(sub(0,2,0) == 9);
            REQUIRE(sub(1,2,0) == 10);
            REQUIRE(sub(2,2,0) == 11);

            REQUIRE(sub(0,0,2) == 33);
            REQUIRE(sub(1,0,2) == 34);
            REQUIRE(sub(2,0,2) == 35);
            
            REQUIRE(sub(0,2,2) == 41);
            REQUIRE(sub(1,2,2) == 42);
            REQUIRE(sub(2,2,2) == 43);
        }
    }

    SECTION("copy_from") {
        // SubVol -> SubVol
        {
            VolumeInt vol({4, 4, 4}, val);
            GpuVolume gpu_vol(vol);

            GpuVolume gpu_dst = gpu_vol({2,4}, {2,4}, {2,4});
            GpuVolume gpu_src = gpu_vol({0,2}, {0,2}, {0,2});
            gpu_dst.copy_from(gpu_src);

            VolumeInt dst = gpu_dst.download();
            VolumeInt src = gpu_src.download();
            for (int z = 0; z < (int)dst.size().z; ++z) {
            for (int y = 0; y < (int)dst.size().y; ++y) {
            for (int x = 0; x < (int)dst.size().x; ++x) {
                REQUIRE(dst(x,y,z) == src(x,y,z));
            }
            }
            }
        }

        // SubVol -> Vol
        {
            VolumeInt vol({4, 4, 4}, val);
            GpuVolume gpu_vol(vol);

            GpuVolume gpu_dst({2,2,2}, stk::Type_Int);
            GpuVolume gpu_src = gpu_vol({0,2}, {0,2}, {0,2});
            gpu_dst.copy_from(gpu_src);

            VolumeInt dst = gpu_dst.download();
            VolumeInt src = gpu_src.download();
            for (int z = 0; z < (int)dst.size().z; ++z) {
            for (int y = 0; y < (int)dst.size().y; ++y) {
            for (int x = 0; x < (int)dst.size().x; ++x) {
                REQUIRE(dst(x,y,z) == src(x,y,z));
            }
            }
            }
        }

        // Vol -> SubVol
        {
            VolumeInt vol({4, 4, 4}, val);
            GpuVolume gpu_vol(vol);

            VolumeInt src = VolumeInt({2,2,2}, sub_val);
            GpuVolume gpu_src(src);
            GpuVolume gpu_dst = gpu_vol({1,3}, {1,3}, {1,3});

            gpu_dst.copy_from(gpu_src);

            VolumeInt dst = gpu_dst.download();
            for (int z = 0; z < (int)dst.size().z; ++z) {
            for (int y = 0; y < (int)dst.size().y; ++y) {
            for (int x = 0; x < (int)dst.size().x; ++x) {
                REQUIRE(dst(x,y,z) == src(x,y,z));
            }
            }
            }
        }
    }
    
    SECTION("upload_download") {
        // [Sub]Volume => GpuVolume::upload
        {
            VolumeInt vol({4, 4, 4}, val);
        
            VolumeInt src(vol, {1, 3}, {1, 3}, {1, 3});
            GpuVolume gpu_sub(src);
            VolumeInt dst = gpu_sub.download();

            for (int z = 0; z < (int)dst.size().z; ++z) {
            for (int y = 0; y < (int)dst.size().y; ++y) {
            for (int x = 0; x < (int)dst.size().x; ++x) {
                REQUIRE(dst(x,y,z) == src(x,y,z));
            }
            }
            }
        }

        // [Sub]Volume => [Sub]GpuVolume::upload
        {
            VolumeInt vol({4, 4, 4}, val);

            VolumeInt src(vol, {2, 4}, {2, 4}, {2, 4});
            GpuVolume gpu_vol(vol);
            GpuVolume gpu_sub(gpu_vol, {2, 4}, {2, 4}, {2, 4});

            VolumeInt dst = gpu_sub.download();
            for (int z = 0; z < (int)dst.size().z; ++z) {
            for (int y = 0; y < (int)dst.size().y; ++y) {
            for (int x = 0; x < (int)dst.size().x; ++x) {
                REQUIRE(dst(x,y,z) == src(x,y,z));
            }
            }
            }
        }

        // Volume      => [Sub]GpuVolume::upload
        {
            VolumeInt vol({4, 4, 4}, val);

            VolumeInt src({2,2,2}, sub_val);
            GpuVolume gpu_vol(vol);
            GpuVolume gpu_sub(gpu_vol, {2, 4}, {2, 4}, {2, 4});
            gpu_sub.upload(src);

            VolumeInt dst = gpu_sub.download();
            for (int z = 0; z < (int)dst.size().z; ++z) {
            for (int y = 0; y < (int)dst.size().y; ++y) {
            for (int x = 0; x < (int)dst.size().x; ++x) {
                REQUIRE(dst(x,y,z) == src(x,y,z));
            }
            }
            }
        }

        // GpuVolume::download => [Sub]Volume
        {
            VolumeInt vol({4, 4, 4}, val);

            VolumeInt dst(vol, {1, 3}, {1, 3}, {1, 3});
            VolumeInt src({2,2,2}, sub_val);
            GpuVolume gpu_vol(src);

            gpu_vol.download(dst);

            for (int z = 0; z < (int)dst.size().z; ++z) {
            for (int y = 0; y < (int)dst.size().y; ++y) {
            for (int x = 0; x < (int)dst.size().x; ++x) {
                REQUIRE(dst(x,y,z) == src(x,y,z));
            }
            }
            }
        }

        // [Sub]GpuVolume::download => [Sub]Volume
        {
            VolumeInt vol({4, 4, 4}, val);
            VolumeInt src(vol, {0, 2}, {0, 2}, {0, 2});
            
            GpuVolume gpu_vol(vol);
            GpuVolume gpu_sub(gpu_vol, {0, 2}, {0, 2}, {0, 2});
            
            VolumeInt dst(vol, {2, 4}, {2, 4}, {2, 4});
            gpu_sub.download(dst);

            for (int z = 0; z < (int)dst.size().z; ++z) {
            for (int y = 0; y < (int)dst.size().y; ++y) {
            for (int x = 0; x < (int)dst.size().x; ++x) {
                REQUIRE(dst(x,y,z) == src(x,y,z));
            }
            }
            }
        }

        // [Sub]GpuVolume::download => Volume
        {
            VolumeInt vol({4, 4, 4}, val);
            VolumeInt src(vol, {0, 2}, {0, 2}, {0, 2});
            
            GpuVolume gpu_vol(vol);
            GpuVolume gpu_sub(gpu_vol, {0, 2}, {0, 2}, {0, 2});
            
            VolumeInt dst({2,2,2});
            gpu_sub.download(dst);

            for (int z = 0; z < (int)dst.size().z; ++z) {
            for (int y = 0; y < (int)dst.size().y; ++y) {
            for (int x = 0; x < (int)dst.size().x; ++x) {
                REQUIRE(dst(x,y,z) == src(x,y,z));
            }
            }
            }
        }
    }
    SECTION("referencing") {
        // Just to check that they actually reference the same memory

        VolumeInt vol({4, 4, 4}, val);
        GpuVolume gpu_vol(vol);
        GpuVolume gpu_sub(gpu_vol, {0,2}, {0, 2}, {0, 2});

        VolumeInt subvol({2,2,2}, -1);
        GpuVolume gpu_sub2(subvol);

        gpu_sub.copy_from(gpu_sub2);

        VolumeInt test = gpu_vol.download();

        for (int z = 0; z < 4; ++z) {
        for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            if (x < 2 && y < 2 && z < 2) {
                REQUIRE(test(x,y,z) == -1);
            }
            else {
                REQUIRE(test(x,y,z) != -1);
            }
        }
        }
        }
    }
}

