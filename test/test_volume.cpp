#include "catch.hpp"

#include <stk/common/error.h>
#include <stk/volume/volume.h>

using namespace stk;

TEST_CASE("volume", "[volume]")
{
    const int W = 2;
    const int H = 3;
    const int D = 4;
    uint8_t test_data[W*H*D];

    for (uint32_t z = 0; z < D; ++z) {
        for (uint32_t y = 0; y < H; ++y) {
            for (uint32_t x = 0; x < W; ++x) {
                test_data[H*W*z + W*y + x] 
                    = uint8_t(H*W*z + W*y + x);
            }
        }
    }
    
    SECTION("constructor")
    {
        Volume vol({W,H,D}, Type_UChar, test_data);
        REQUIRE(vol.size() == dim3{W, H, D});
        REQUIRE(vol.valid());
        REQUIRE(vol.ptr());

        for (uint32_t z = 0; z < D; ++z) {
            for (uint32_t y = 0; y < H; ++y) {
                for (uint32_t x = 0; x < W; ++x) {
                    REQUIRE(static_cast<uint8_t*>(vol.ptr())[H*W*z + W*y + x]
                            == uint8_t(H*W*z + W*y + x));
                }
            }
        }
    }
    SECTION("allocate")
    {
        Volume vol;
        vol.allocate({W,H,D}, Type_UChar);
        REQUIRE(vol.size() == dim3{W, H, D});
        REQUIRE(vol.valid());
        REQUIRE(vol.ptr());

        memcpy(vol.ptr(), test_data, W*H*D);
            
        for (uint32_t z = 0; z < D; ++z) {
            for (uint32_t y = 0; y < H; ++y) {
                for (uint32_t x = 0; x < W; ++x) {
                    REQUIRE(static_cast<uint8_t*>(vol.ptr())[H*W*z + W*y + x]
                            == uint8_t(H*W*z + W*y + x));
                }
            }
        }
    }
    SECTION("release")
    {
        Volume vol;
        vol.allocate({W,H,D}, Type_UChar);
        REQUIRE(vol.size() == dim3{W, H, D});
        REQUIRE(vol.valid());
        REQUIRE(vol.ptr());

        vol.release();
        REQUIRE(!vol.valid());
    }
}
TEST_CASE("volume_ref", "[volume]")
{
    // Test reference handling

    const int W = 2;
    const int H = 3;
    const int D = 4;

    uint8_t test_data[W*H*D];
    for (int i = 0; i < W*H*D; ++i)
        test_data[i] = uint8_t(i);

    Volume vol({W,H,D}, Type_UChar, test_data);

    SECTION("assignment")
    {
        // Create a soft copy, will be referencing the same memory
        Volume copy = vol;
        REQUIRE(copy.valid());
        REQUIRE(copy.ptr() == vol.ptr()); // Should reference same memory

        // Releasing the old volume should not affect the new copys access
        //  to the data
        vol.release();
        REQUIRE(!vol.valid());
        REQUIRE(copy.valid());

        for (int i = 0; i < W*H*D; ++i) {
            REQUIRE(static_cast<uint8_t*>(copy.ptr())[i] == test_data[i]);
        }
    }
    SECTION("copy_constructor")
    {
        // Create a soft copy, will be referencing the same memory
        Volume copy(vol);
        REQUIRE(copy.valid());
        REQUIRE(copy.ptr() == vol.ptr()); // Should reference same memory

        // Releasing the old volume should not affect the new copys access
        //  to the data
        vol.release();
        REQUIRE(!vol.valid());
        REQUIRE(copy.valid());

        for (int i = 0; i < W*H*D; ++i) {
            REQUIRE(static_cast<uint8_t*>(copy.ptr())[i] == test_data[i]);
        }
    }
}

TEST_CASE("volume_types", "[volume]")
{
    const int W = 2;
    const int H = 3;
    const int D = 4;

    #define TYPE_TEST(T, T_id) \
        SECTION(#T) \
        { \
            T test_data[W*H*D]; \
            for (int i = 0; i < W*H*D; ++i) { \
                test_data[i] = T{0}; \
            } \
            Volume vol({W,H,D}, T_id, test_data); \
            REQUIRE(vol.valid()); \
        }

    TYPE_TEST(uint8_t,  Type_UChar);
    TYPE_TEST(uchar2,   Type_UChar2);
    TYPE_TEST(uchar3,   Type_UChar3);
    TYPE_TEST(uchar4,   Type_UChar4);
    TYPE_TEST(float,    Type_Float);
    TYPE_TEST(float2,   Type_Float2);
    TYPE_TEST(float3,   Type_Float3);
    TYPE_TEST(float4,   Type_Float4);
    TYPE_TEST(double,   Type_Double);
    TYPE_TEST(double2,  Type_Double2);
    TYPE_TEST(double3,  Type_Double3);
    TYPE_TEST(double4,  Type_Double4);

    #undef TYPE_TEST
}

TEST_CASE("volume_meta_data", "[volume]")
{
    Volume vol({4,4,4}, Type_Float);

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

TEST_CASE("volume_clone", "[volume]")
{
    const int W = 2;
    const int H = 3;
    const int D = 4;

    float test_data[W*H*D];
    for (int i = 0; i < W*H*D; ++i)
        test_data[i] = float(i);

    Volume vol({W,H,D}, Type_Float, test_data);
    vol.set_origin({2.0f, 3.0f, 4.0f});
    vol.set_spacing({5.0f, 6.0f, 7.0f});
    REQUIRE(vol.valid());

    Volume clone = vol.clone();
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

    for (int i = 0; i < W*H*D; ++i) {
        REQUIRE(static_cast<float*>(clone.ptr())[i] == Approx(test_data[i]));
    }

    vol.release();
    REQUIRE(!vol.valid());
    REQUIRE(clone.valid());
}

TEST_CASE("volume_copy_from", "[volume]")
{
    const int W = 2;
    const int H = 3;
    const int D = 4;

    float test_data[W*H*D];
    for (int i = 0; i < W*H*D; ++i)
        test_data[i] = float(i);

    Volume vol({W,H,D}, Type_Float, test_data);
    vol.set_origin({2.0f, 3.0f, 4.0f});
    vol.set_spacing({5.0f, 6.0f, 7.0f});

    Volume copy({W,H,D}, Type_Float);
    copy.copy_from(vol);

    REQUIRE(copy.ptr() != vol.ptr()); // Should not point to same memory

    // copy_from should also copy meta data
    REQUIRE(copy.origin().x == Approx(vol.origin().x));
    REQUIRE(copy.origin().y == Approx(vol.origin().y));
    REQUIRE(copy.origin().z == Approx(vol.origin().z));
    REQUIRE(copy.spacing().x == Approx(vol.spacing().x));
    REQUIRE(copy.spacing().y == Approx(vol.spacing().y));
    REQUIRE(copy.spacing().z == Approx(vol.spacing().z));

    for (int i = 0; i < W*H*D; ++i) {
        REQUIRE(static_cast<float*>(copy.ptr())[i] == Approx(test_data[i]));
    }
}
TEST_CASE("volume_as_type", "[volume]")
{
    const int W = 2;
    const int H = 3;
    const int D = 4;


    // Currently we only support float => double
    SECTION("float_to_double")
    {
        float test_data[W*H*D];
        for (int i = 0; i < W*H*D; ++i)
            test_data[i] = float(i);

        Volume vol({W,H,D}, Type_Float, test_data);
        vol.set_origin({2.0f, 3.0f, 4.0f});
        vol.set_spacing({5.0f, 6.0f, 7.0f});
        
        Volume vol2 = vol.as_type(Type_Double);
        REQUIRE(vol2.valid());
        REQUIRE(vol2.voxel_type() == Type_Double);
        REQUIRE(vol2.ptr() != vol.ptr());

        // Should include meta data
        REQUIRE(vol2.origin().x == Approx(vol.origin().x));
        REQUIRE(vol2.origin().y == Approx(vol.origin().y));
        REQUIRE(vol2.origin().z == Approx(vol.origin().z));
        REQUIRE(vol2.spacing().x == Approx(vol.spacing().x));
        REQUIRE(vol2.spacing().y == Approx(vol.spacing().y));
        REQUIRE(vol2.spacing().z == Approx(vol.spacing().z));
        
        for (int i = 0; i < W*H*D; ++i) {
            REQUIRE(static_cast<double*>(vol2.ptr())[i] == Approx(test_data[i]));
        }
    }
    SECTION("double_to_float")
    {
        double test_data[W*H*D];
        for (int i = 0; i < W*H*D; ++i)
            test_data[i] = double(i);

        Volume vol({W,H,D}, Type_Double, test_data);
        
        Volume vol2 = vol.as_type(Type_Float);
        REQUIRE(vol2.valid());
        REQUIRE(vol2.voxel_type() == Type_Float);
        REQUIRE(vol2.ptr() != vol.ptr());

        for (int i = 0; i < W*H*D; ++i) {
            REQUIRE(static_cast<float*>(vol2.ptr())[i] == Approx(test_data[i]));
        }
    }
    SECTION("double4_to_float4")
    {
        double4 test_data[W*H*D];
        for (int i = 0; i < W*H*D; ++i)
            test_data[i] = double4{double(i), double(i+1), double(i+2), double(i+3)};

        Volume vol({W,H,D}, Type_Double4, test_data);
        
        Volume vol2 = vol.as_type(Type_Float4);
        REQUIRE(vol2.valid());
        REQUIRE(vol2.voxel_type() == Type_Float4);
        REQUIRE(vol2.ptr() != vol.ptr());

        for (int i = 0; i < W*H*D; ++i) {
            float4 s = static_cast<float4*>(vol2.ptr())[i];
            double4 t = test_data[i];
            REQUIRE(s.x == Approx(t.x));
            REQUIRE(s.y == Approx(t.y));
            REQUIRE(s.z == Approx(t.z));
            REQUIRE(s.w == Approx(t.w));
        }
    }
    SECTION("float_to_float")
    {
        float test_data[W*H*D];
        for (int i = 0; i < W*H*D; ++i)
            test_data[i] = float(i);

        Volume vol({W,H,D}, Type_Float, test_data);
        
        // Should return itself
        Volume vol2 = vol.as_type(Type_Float);
        REQUIRE(vol2.valid());
        REQUIRE(vol2.voxel_type() == Type_Float);
        REQUIRE(vol2.ptr() == vol.ptr());
    }
    SECTION("float_to_float2")
    {
        float test_data[W*H*D];
        for (int i = 0; i < W*H*D; ++i)
            test_data[i] = float(i);

        Volume vol({W,H,D}, Type_Float, test_data);
        
        // Cannot convert from float to float2
        REQUIRE_THROWS_AS(vol.as_type(Type_Float2), FatalException);
    }
    SECTION("float_to_uchar")
    {
        float test_data[W*H*D];
        for (int i = 0; i < W*H*D; ++i)
            test_data[i] = float(i);

        Volume vol({W,H,D}, Type_Float, test_data);
        
        // Not implemented
        REQUIRE_THROWS_AS(vol.as_type(Type_UChar), FatalException);
    }
    // ...
}
