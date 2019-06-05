#include "catch.hpp"

#include <cstring>
#include <stk/common/error.h>
#include <stk/image/volume.h>

using namespace stk;

namespace {
    const uint32_t W = 2;
    const uint32_t H = 3;
    const uint32_t D = 4;
}

TEST_CASE("volume", "[volume]")
{
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

    uint8_t test_data[W*H*D];
    for (uint32_t i = 0; i < W*H*D; ++i)
        test_data[i] = uint8_t(i);

    SECTION("assignment")
    {
        Volume vol({W,H,D}, Type_UChar, test_data);

        // Create a soft copy, will be referencing the same memory
        Volume copy = vol;
        REQUIRE(copy.valid());
        REQUIRE(copy.ptr() == vol.ptr()); // Should reference same memory

        // Releasing the old volume should not affect the new copys access
        //  to the data
        vol.release();
        REQUIRE(!vol.valid());
        REQUIRE(copy.valid());

        for (uint32_t i = 0; i < W*H*D; ++i) {
            REQUIRE(static_cast<uint8_t*>(copy.ptr())[i] == test_data[i]);
        }
    }
    SECTION("copy_constructor")
    {
        Volume vol({W,H,D}, Type_UChar, test_data);

        // Create a soft copy, will be referencing the same memory
        Volume copy(vol);
        REQUIRE(copy.valid());
        REQUIRE(copy.ptr() == vol.ptr()); // Should reference same memory

        // Releasing the old volume should not affect the new copys access
        //  to the data
        vol.release();
        REQUIRE(!vol.valid());
        REQUIRE(copy.valid());

        for (uint32_t i = 0; i < W*H*D; ++i) {
            REQUIRE(static_cast<uint8_t*>(copy.ptr())[i] == test_data[i]);
        }
    }
}

TEST_CASE("volume_types", "[volume]")
{
    #define TYPE_TEST(T, T_id) \
        SECTION(#T) \
        { \
            T test_data[W*H*D]; \
            for (uint32_t i = 0; i < W*H*D; ++i) { \
                test_data[i] = T{}; \
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

TEST_CASE("volume_copy_meta", "[volume]")
{
    Volume a({4,4,4}, Type_Float);

    a.set_origin({2.0f, 3.0f, 4.0f});
    a.set_spacing({5.0f, 6.0f, 7.0f});

    Volume b({2,2,2}, Type_Float);
    b.copy_meta_from(a);

    REQUIRE(b.origin().x == Approx(2.0f));
    REQUIRE(b.origin().y == Approx(3.0f));
    REQUIRE(b.origin().z == Approx(4.0f));

    REQUIRE(b.spacing().x == Approx(5.0f));
    REQUIRE(b.spacing().y == Approx(6.0f));
    REQUIRE(b.spacing().z == Approx(7.0f));
}

TEST_CASE("volume_coordinate_conversion", "[volume]")
{
    stk::Volume vol = stk::Volume({1, 1, 1}, stk::Type_Float);

    // Random numbers
    vol.set_origin({0.11450715f, 0.69838153f, 0.46470744f});
    vol.set_spacing({0.51708509f, 0.93414316f, 0.38919406f});
    vol.set_direction({
        0.49132562f, 0.8060089f , 0.12580945f,
        0.25510848f, 0.96823085f, 0.42032331f,
        0.20022329f, 0.44585017f, 0.70957238f,
    });

    float3 point = vol.index2point(float3({10.4f, 12.7f, 17.1f}));

    // oracle from SimpleITK
    CHECK(point.x == Approx(13.15617270574316f));
    CHECK(point.y == Approx(16.35433906555294f));
    CHECK(point.z == Approx(11.55320054939331f));

    point = vol.index2point(int3({10, 12, 17}));

    // oracle from SimpleITK
    CHECK(point.x == Approx(12.522604025121286f));
    CHECK(point.y == Approx(15.65208885738362f));
    CHECK(point.z == Approx(11.192629901994575f));

    float3 index = vol.point2index(float3({130.3f, -19.2f, 55.7f}));

    // oracle from SimpleITK
    CHECK(index.x == Approx(1073.011446554096f));
    CHECK(index.y == Approx(-195.14144294668f));
    CHECK(index.z == Approx(92.03969103390264f));

    index = vol.point2index(int3({130, -19, 55}));

    // oracle from SimpleITK
    CHECK(index.x == Approx(1068.0339658527405f));
    CHECK(index.y == Approx(-193.67283470127222f));
    CHECK(index.z == Approx(89.15613539524472f));
}

TEST_CASE("volume_clone", "[volume]")
{
    float test_data[W*H*D];
    for (uint32_t i = 0; i < W*H*D; ++i)
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

    for (uint32_t i = 0; i < W*H*D; ++i) {
        REQUIRE(static_cast<float*>(clone.ptr())[i] == Approx(test_data[i]));
    }

    vol.release();
    REQUIRE(!vol.valid());
    REQUIRE(clone.valid());
}

TEST_CASE("volume_copy_from", "[volume]")
{
    float test_data[W*H*D];
    for (uint32_t i = 0; i < W*H*D; ++i)
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

    for (uint32_t i = 0; i < W*H*D; ++i) {
        REQUIRE(static_cast<float*>(copy.ptr())[i] == Approx(test_data[i]));
    }
}
TEST_CASE("volume_as_type", "[volume]")
{
    // Currently we only support float => double
    SECTION("float_to_double")
    {
        float test_data[W*H*D];
        for (uint32_t i = 0; i < W*H*D; ++i)
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

        for (uint32_t i = 0; i < W*H*D; ++i) {
            REQUIRE(static_cast<double*>(vol2.ptr())[i] == Approx(test_data[i]));
        }
    }
    SECTION("double_to_float")
    {
        double test_data[W*H*D];
        for (uint32_t i = 0; i < W*H*D; ++i)
            test_data[i] = double(i);

        Volume vol({W,H,D}, Type_Double, test_data);

        Volume vol2 = vol.as_type(Type_Float);
        REQUIRE(vol2.valid());
        REQUIRE(vol2.voxel_type() == Type_Float);
        REQUIRE(vol2.ptr() != vol.ptr());

        for (uint32_t i = 0; i < W*H*D; ++i) {
            REQUIRE(static_cast<float*>(vol2.ptr())[i] == Approx(test_data[i]));
        }
    }
    SECTION("double4_to_float4")
    {
        double4 test_data[W*H*D];
        for (uint32_t i = 0; i < W*H*D; ++i)
            test_data[i] = double4{double(i), double(i+1), double(i+2), double(i+3)};

        Volume vol({W,H,D}, Type_Double4, test_data);

        Volume vol2 = vol.as_type(Type_Float4);
        REQUIRE(vol2.valid());
        REQUIRE(vol2.voxel_type() == Type_Float4);
        REQUIRE(vol2.ptr() != vol.ptr());

        for (uint32_t i = 0; i < W*H*D; ++i) {
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
        for (uint32_t i = 0; i < W*H*D; ++i)
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
        for (uint32_t i = 0; i < W*H*D; ++i)
            test_data[i] = float(i);

        Volume vol({W,H,D}, Type_Float, test_data);

        // Cannot convert from float to float2
        REQUIRE_THROWS_AS(vol.as_type(Type_Float2), FatalException);
    }
    SECTION("float_to_uchar")
    {
        float test_data[W*H*D];
        for (uint32_t i = 0; i < W*H*D; ++i)
            test_data[i] = float(i);

        Volume vol({W,H,D}, Type_Float, test_data);

        // Not implemented
        REQUIRE_THROWS_AS(vol.as_type(Type_UChar), FatalException);
    }
    // ...
}
TEST_CASE("volume_helper", "[volume]")
{
    SECTION("constructor")
    {
        VolumeHelper<float> vol;
        REQUIRE(!vol.valid()); // Shouldn't have allocated anything
        REQUIRE(vol.voxel_type() == Type_Float); // However, type should be set
    }
    SECTION("constructor2")
    {
        VolumeHelper<float> vol({W,H,D});
        REQUIRE(vol.valid());
        REQUIRE(vol.voxel_type() == Type_Float);

        for (uint32_t i = 0; i < W*H*D; ++i) {
            static_cast<float*>(vol.ptr())[i] = float(i);
        }
    }
    SECTION("constructor3")
    {
        VolumeHelper<float> vol({W,H,D}, 3.0f);
        REQUIRE(vol.valid());
        REQUIRE(vol.voxel_type() == Type_Float);

        for (uint32_t i = 0; i < W*H*D; ++i) {
            REQUIRE(static_cast<float*>(vol.ptr())[i] == Approx(3.0f));
        }
    }


    SECTION("copy_constructor")
    {
        float test_data[W*H*D];
        for (uint32_t i = 0; i < W*H*D; ++i)
            test_data[i] = float(i);

        Volume src({W,H,D}, Type_Float, test_data);

        // Without conversion
        VolumeHelper<float> copy1(src);
        REQUIRE(copy1.valid());
        REQUIRE(copy1.voxel_type() == Type_Float);
        REQUIRE(copy1.size() == dim3{W,H,D});

        for (uint32_t i = 0; i < W*H*D; ++i) {
            REQUIRE(static_cast<float*>(copy1.ptr())[i] == Approx(test_data[i]));
        }

        // With conversion
        VolumeHelper<double> copy2(src);
        REQUIRE(copy2.valid());
        REQUIRE(copy2.voxel_type() == Type_Double);
        REQUIRE(copy2.size() == dim3{W,H,D});

        for (uint32_t i = 0; i < W*H*D; ++i) {
            REQUIRE(static_cast<double*>(copy2.ptr())[i] == Approx(test_data[i]));
        }

        // With invalid conversion
        REQUIRE_THROWS_AS(VolumeHelper<float3>(src), FatalException);
        REQUIRE_THROWS_AS(VolumeHelper<double3>(src), FatalException);
    }
    SECTION("copy_assignment")
    {
        double4 test_data[W*H*D];
        for (uint32_t i = 0; i < W*H*D; ++i)
            test_data[i] = double4{double(i), double(i+1), double(i+2), double(i+3)};

        Volume src({W,H,D}, Type_Double4, test_data);

        // Without conversion
        VolumeHelper<double4> copy1(src);
        REQUIRE(copy1.valid());
        REQUIRE(copy1.voxel_type() == Type_Double4);
        REQUIRE(copy1.size() == dim3{W,H,D});

        for (uint32_t i = 0; i < W*H*D; ++i) {
            double4 s = static_cast<double4*>(copy1.ptr())[i];
            double4 t = test_data[i];
            REQUIRE(s.x == Approx(t.x));
            REQUIRE(s.y == Approx(t.y));
            REQUIRE(s.z == Approx(t.z));
            REQUIRE(s.w == Approx(t.w));
        }

        // With conversion
        VolumeHelper<float4> copy2(src);
        REQUIRE(copy2.valid());
        REQUIRE(copy2.voxel_type() == Type_Float4);
        REQUIRE(copy2.size() == dim3{W,H,D});

        for (uint32_t i = 0; i < W*H*D; ++i) {
            float4 s = static_cast<float4*>(copy2.ptr())[i];
            double4 t = test_data[i];
            REQUIRE(s.x == Approx(t.x));
            REQUIRE(s.y == Approx(t.y));
            REQUIRE(s.z == Approx(t.z));
            REQUIRE(s.w == Approx(t.w));
        }

        // With invalid conversion
        REQUIRE_THROWS_AS(VolumeHelper<float2>(src), FatalException);
        REQUIRE_THROWS_AS(VolumeHelper<double2>(src), FatalException);
    }
    SECTION("indexing")
    {
        uchar3 test_data[W*H*D];
        for (uint32_t i = 0; i < W*H*D; ++i)
            test_data[i] = uchar3{uint8_t(i), uint8_t(i+1), uint8_t(i+2)};

        VolumeHelper<uchar3> vol({W,H,D}, test_data);
        for (uint32_t z = 0; z < D; ++z) {
            for (uint32_t y = 0; y < H; ++y) {
                for (uint32_t x = 0; x < W; ++x) {
                    uchar3 s = vol(x,y,z);
                    uchar3 t = test_data[x + y * W + z * W * H];
                    REQUIRE(s.x == t.x);
                    REQUIRE(s.y == t.y);
                    REQUIRE(s.z == t.z);
                }
            }
        }

    }
    SECTION("offset")
    {
        VolumeHelper<float> vol({W,H,D});
        REQUIRE(vol.offset(0,0,0) == 0);
        REQUIRE(vol.offset(1,0,0) == sizeof(float));
        REQUIRE(vol.offset(0,1,0) == W*sizeof(float));
        REQUIRE(vol.offset(0,0,1) == W*H*sizeof(float));
    }
    SECTION("fill")
    {
        VolumeHelper<float> vol({W,H,D});
        vol.fill(5.5f);
        for (uint32_t i = 0; i < W*H*D; ++i) {
            REQUIRE(static_cast<float*>(vol.ptr())[i] == Approx(5.5f));
        }
    }
    SECTION("at")
    {
        VolumeHelper<float> vol({W,H,D}, 7.0f);

        REQUIRE(vol.at(0,0,0, Border_Constant) == Approx(7.0f));
        REQUIRE(vol.at(1,1,1, Border_Replicate) == Approx(7.0f));

        // Border_Constant should at zeros at the border
        REQUIRE(vol.at(-1,0,0, Border_Constant) == Approx(0.0f));
        REQUIRE(vol.at(0,-1,0, Border_Constant) == Approx(0.0f));
        REQUIRE(vol.at(0,0,-1, Border_Constant) == Approx(0.0f));

        REQUIRE(vol.at(W,0,0, Border_Constant) == Approx(0.0f));
        REQUIRE(vol.at(0,H,0, Border_Constant) == Approx(0.0f));
        REQUIRE(vol.at(0,0,D, Border_Constant) == Approx(0.0f));

        // Border_Replicate replicates the value closest to the border of the volume

        REQUIRE(vol.at(-1,0,0, Border_Replicate) == Approx(7.0f));
        REQUIRE(vol.at(0,-1,0, Border_Replicate) == Approx(7.0f));
        REQUIRE(vol.at(0,0,-1, Border_Replicate) == Approx(7.0f));

        REQUIRE(vol.at(W,0,0, Border_Replicate) == Approx(7.0f));
        REQUIRE(vol.at(0,H,0, Border_Replicate) == Approx(7.0f));
        REQUIRE(vol.at(0,0,D, Border_Replicate) == Approx(7.0f));
    }
    SECTION("linear_at")
    {
        VolumeHelper<float> vol({W,H,D}, 8.0f);

        // Identical behaviour to at() for integers

        REQUIRE(vol.linear_at(0,0,0, Border_Constant) == Approx(8.0f));
        REQUIRE(vol.linear_at(1,1,1, Border_Replicate) == Approx(8.0f));

        // Border_Constant should at zeros at the border
        REQUIRE(vol.linear_at(-1,0,0, Border_Constant) == Approx(0.0f));
        REQUIRE(vol.linear_at(0,-1,0, Border_Constant) == Approx(0.0f));
        REQUIRE(vol.linear_at(0,0,-1, Border_Constant) == Approx(0.0f));

        REQUIRE(vol.linear_at(float{W},0,0, Border_Constant) == Approx(0.0f));
        REQUIRE(vol.linear_at(0,float{H},0, Border_Constant) == Approx(0.0f));
        REQUIRE(vol.linear_at(0,0,float{D}, Border_Constant) == Approx(0.0f));

        // Border_Replicate replicates the value closest to the border of the volume

        REQUIRE(vol.linear_at(-1,0,0, Border_Replicate) == Approx(8.0f));
        REQUIRE(vol.linear_at(0,-1,0, Border_Replicate) == Approx(8.0f));
        REQUIRE(vol.linear_at(0,0,-1, Border_Replicate) == Approx(8.0f));

        REQUIRE(vol.linear_at(float{W},0,0, Border_Replicate) == Approx(8.0f));
        REQUIRE(vol.linear_at(0,float{H},0, Border_Replicate) == Approx(8.0f));
        REQUIRE(vol.linear_at(0,0,float{D}, Border_Replicate) == Approx(8.0f));

        // Simple tests for the linear interpolation
        // Given data [2, 4], value at i=0.5 should be 3
        {
            #define LINEAR_AT_TEST(T) \
                { \
                    T v_data[] = {T{2}, T{4}}; \
                    VolumeHelper<T> v_x({2,1,1}, v_data); \
                    VolumeHelper<T> v_y({1,2,1}, v_data); \
                    VolumeHelper<T> v_z({1,1,2}, v_data); \
                    REQUIRE(v_x.linear_at({0.5, 0, 0}, Border_Constant) == Approx(3)); \
                    REQUIRE(v_y.linear_at({0, 0.5, 0}, Border_Constant) == Approx(3)); \
                    REQUIRE(v_z.linear_at({0, 0, 0.5}, Border_Constant) == Approx(3)); \
                }

            LINEAR_AT_TEST(float);
            LINEAR_AT_TEST(double);
            LINEAR_AT_TEST(uint8_t);
        }
    }
}
TEST_CASE("find_min_max", "[volume]")
{
    SECTION("float") {
        float val[] = {
            2, 1,
            3, 7,
            3, 7,

            8, 7,
            3, 2,
            1, 2,

            3, 2,
            1, 2,
            4, 5,

            3, 2,
            1, 2,
            4, 5
        };

        VolumeHelper<float> vol({2,3,4}, val);

        float min, max;
        find_min_max(vol, min, max);

        REQUIRE(min == Approx(1));
        REQUIRE(max == Approx(8));
    }
    SECTION("double") {
        double val[] = {
            2, 1,
            3, 7,
            3, 7,

            8, 7,
            3, 2,
            1, 2,

            3, 2,
            1, 2,
            4, 5,

            3, 2,
            1, 2,
            4, 5
        };

        VolumeHelper<double> vol({2,3,4}, val);

        double min, max;
        find_min_max(vol, min, max);

        REQUIRE(min == Approx(1));
        REQUIRE(max == Approx(8));
    }
    SECTION("short") {
        short val[] = {
            2, 1,
            3, 7,
            3, 7,

            8, 7,
            3, 2,
            1, 2,

            3, 2,
            1, 2,
            4, 5,

            3, 2,
            1, 2,
            4, 5
        };

        VolumeHelper<short> vol({2,3,4}, val);

        short min, max;
        find_min_max(vol, min, max);

        REQUIRE(min == Approx(1));
        REQUIRE(max == Approx(8));
    }
    SECTION("ushort") {
        uint16_t val[] = {
            2, 1,
            3, 7,
            3, 7,

            8, 7,
            3, 2,
            1, 2,

            3, 2,
            1, 2,
            4, 5,

            3, 2,
            1, 2,
            4, 5
        };

        VolumeHelper<uint16_t> vol({2,3,4}, val);

        uint16_t min, max;
        find_min_max(vol, min, max);

        REQUIRE(min == Approx(1));
        REQUIRE(max == Approx(8));
    }
}
TEST_CASE("volume_region", "[volume]")
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

    SECTION("constructor") {
        VolumeInt vol({4, 4, 4}, val);
        vol.set_origin(float3{10.0f, 20.0f, 30.0f});
        vol.set_spacing(float3{1.0f, 2.0f, 3.0f});

        VolumeInt sub(vol, {1,4}, {1,4}, {1, 4});
        REQUIRE(sub.size().x == 3);
        REQUIRE(sub.size().y == 3);
        REQUIRE(sub.size().z == 3);
        REQUIRE(sub.is_contiguous() == false);

        REQUIRE(sub.origin().x == Approx(11.0f));
        REQUIRE(sub.origin().y == Approx(22.0f));
        REQUIRE(sub.origin().z == Approx(33.0f));

        REQUIRE(sub.spacing().x == Approx(1.0f));
        REQUIRE(sub.spacing().y == Approx(2.0f));
        REQUIRE(sub.spacing().z == Approx(3.0f));

        REQUIRE(sub(0,0,0) == 22);
        REQUIRE(sub(1,0,0) == 23);
        REQUIRE(sub(0,1,0) == 26);
        REQUIRE(sub(1,1,0) == 27);
        REQUIRE(sub(0,0,1) == 38);
        REQUIRE(sub(1,0,1) == 39);
        REQUIRE(sub(0,1,1) == 42);
        REQUIRE(sub(1,1,1) == 43);

        VolumeInt sub2(sub, {1,2}, {1,2}, {0,2});
        REQUIRE(sub2.size().x == 1);
        REQUIRE(sub2.size().y == 1);
        REQUIRE(sub2.size().z == 2);
        REQUIRE(sub2.is_contiguous() == false);

        REQUIRE(sub2.origin().x == Approx(12.0f));
        REQUIRE(sub2.origin().y == Approx(24.0f));
        REQUIRE(sub2.origin().z == Approx(33.0f));

        REQUIRE(sub2(0,0,0) == 27);
        REQUIRE(sub2(0,0,1) == 43);

        VolumeInt sub3(vol, {0,3}, {0,3}, {0,3});
        REQUIRE(sub3.size().x == 3);
        REQUIRE(sub3.size().y == 3);
        REQUIRE(sub3.size().z == 3);
        REQUIRE(sub3.is_contiguous() == false);

        REQUIRE(sub3.origin().x == Approx(10.0f));
        REQUIRE(sub3.origin().y == Approx(20.0f));
        REQUIRE(sub3.origin().z == Approx(30.0f));

        REQUIRE(sub3(0,0,0) == 1);
        REQUIRE(sub3(1,0,0) == 2);
        REQUIRE(sub3(2,0,0) == 3);

        REQUIRE(sub3(0,2,0) == 9);
        REQUIRE(sub3(1,2,0) == 10);
        REQUIRE(sub3(2,2,0) == 11);

        REQUIRE(sub3(0,0,2) == 33);
        REQUIRE(sub3(1,0,2) == 34);
        REQUIRE(sub3(2,0,2) == 35);

        REQUIRE(sub3(0,2,2) == 41);
        REQUIRE(sub3(1,2,2) == 42);
        REQUIRE(sub3(2,2,2) == 43);
    }
    SECTION("copy_from") {
        // SubVol -> SubVol
        {
            VolumeInt vol({4, 4, 4}, val);
            VolumeInt dst = vol({2,4}, {2,4}, {2,4});
            VolumeInt src = vol({0,2}, {0,2}, {0,2});
            dst.copy_from(src);

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
            VolumeInt dst = VolumeInt({2,2,2});
            VolumeInt src = vol({0,2}, {0,2}, {0,2});

            dst.copy_from(src);

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

            int sub_val[] = {
                1, 2,
                3, 4,

                5, 6,
                7, 8
            };

            VolumeInt dst = vol({1,3}, {1,3}, {1,3});
            VolumeInt src = VolumeInt({2,2,2}, sub_val);

            dst.copy_from(src);

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
        VolumeInt subvol(vol, {0,2}, {0, 2}, {0, 2});

        for (int z = 0; z < 2; ++z) {
        for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            subvol(x,y,z) = -1;
        }
        }
        }

        for (int z = 0; z < 4; ++z) {
        for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            if (x < 2 && y < 2 && z < 2) {
                REQUIRE(vol(x,y,z) == -1);
            }
            else {
                REQUIRE(vol(x,y,z) != -1);
            }
        }
        }
        }
    }
}
