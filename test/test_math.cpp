#include "catch.hpp"

#include <stk/math/float3.h>
#include <stk/math/float4.h>
#include <stk/math/int3.h>
#include <stk/math/math.h>
#include <stk/math/matrix3x3f.h>

TEST_CASE("math", "[math]")
{
    // char
    REQUIRE(char2{1,2} == char2{1,2});
    REQUIRE(char2{1,2} != char2{3,2});
    REQUIRE(char2{1,2} != char2{1,3});

    REQUIRE(char3{1,2,3} == char3{1,2,3});
    REQUIRE(char3{1,2,3} != char3{4,2,3});
    REQUIRE(char3{1,2,3} != char3{1,4,3});
    REQUIRE(char3{1,2,3} != char3{1,2,4});

    REQUIRE(char4{1,2,3,4} == char4{1,2,3,4});
    REQUIRE(char4{1,2,3,4} != char4{5,2,3,4});
    REQUIRE(char4{1,2,3,4} != char4{1,5,3,4});
    REQUIRE(char4{1,2,3,4} != char4{1,2,5,4});
    REQUIRE(char4{1,2,3,4} != char4{1,2,3,5});

    // uchar
    REQUIRE(uchar2{1,2} == uchar2{1,2});
    REQUIRE(uchar2{1,2} != uchar2{3,2});
    REQUIRE(uchar2{1,2} != uchar2{1,3});

    REQUIRE(uchar3{1,2,3} == uchar3{1,2,3});
    REQUIRE(uchar3{1,2,3} != uchar3{4,2,3});
    REQUIRE(uchar3{1,2,3} != uchar3{1,4,3});
    REQUIRE(uchar3{1,2,3} != uchar3{1,2,4});

    REQUIRE(uchar4{1,2,3,4} == uchar4{1,2,3,4});
    REQUIRE(uchar4{1,2,3,4} != uchar4{5,2,3,4});
    REQUIRE(uchar4{1,2,3,4} != uchar4{1,5,3,4});
    REQUIRE(uchar4{1,2,3,4} != uchar4{1,2,5,4});
    REQUIRE(uchar4{1,2,3,4} != uchar4{1,2,3,5});

    // short
    REQUIRE(short2{1,2} == short2{1,2});
    REQUIRE(short2{1,2} != short2{3,2});
    REQUIRE(short2{1,2} != short2{1,3});

    REQUIRE(short3{1,2,3} == short3{1,2,3});
    REQUIRE(short3{1,2,3} != short3{4,2,3});
    REQUIRE(short3{1,2,3} != short3{1,4,3});
    REQUIRE(short3{1,2,3} != short3{1,2,4});

    REQUIRE(short4{1,2,3,4} == short4{1,2,3,4});
    REQUIRE(short4{1,2,3,4} != short4{5,2,3,4});
    REQUIRE(short4{1,2,3,4} != short4{1,5,3,4});
    REQUIRE(short4{1,2,3,4} != short4{1,2,5,4});
    REQUIRE(short4{1,2,3,4} != short4{1,2,3,5});

    // ushort
    REQUIRE(ushort2{1,2} == ushort2{1,2});
    REQUIRE(ushort2{1,2} != ushort2{3,2});
    REQUIRE(ushort2{1,2} != ushort2{1,3});

    REQUIRE(ushort3{1,2,3} == ushort3{1,2,3});
    REQUIRE(ushort3{1,2,3} != ushort3{4,2,3});
    REQUIRE(ushort3{1,2,3} != ushort3{1,4,3});
    REQUIRE(ushort3{1,2,3} != ushort3{1,2,4});

    REQUIRE(ushort4{1,2,3,4} == ushort4{1,2,3,4});
    REQUIRE(ushort4{1,2,3,4} != ushort4{5,2,3,4});
    REQUIRE(ushort4{1,2,3,4} != ushort4{1,5,3,4});
    REQUIRE(ushort4{1,2,3,4} != ushort4{1,2,5,4});
    REQUIRE(ushort4{1,2,3,4} != ushort4{1,2,3,5});

    // int
    REQUIRE(int2{1,2} == int2{1,2});
    REQUIRE(int2{1,2} != int2{3,2});
    REQUIRE(int2{1,2} != int2{1,3});

    REQUIRE(int3{1,2,3} == int3{1,2,3});
    REQUIRE(int3{1,2,3} != int3{4,2,3});
    REQUIRE(int3{1,2,3} != int3{1,4,3});
    REQUIRE(int3{1,2,3} != int3{1,2,4});

    REQUIRE(int4{1,2,3,4} == int4{1,2,3,4});
    REQUIRE(int4{1,2,3,4} != int4{5,2,3,4});
    REQUIRE(int4{1,2,3,4} != int4{1,5,3,4});
    REQUIRE(int4{1,2,3,4} != int4{1,2,5,4});
    REQUIRE(int4{1,2,3,4} != int4{1,2,3,5});

    // uint
    REQUIRE(uint2{1,2} == uint2{1,2});
    REQUIRE(uint2{1,2} != uint2{3,2});
    REQUIRE(uint2{1,2} != uint2{1,3});

    REQUIRE(uint3{1,2,3} == uint3{1,2,3});
    REQUIRE(uint3{1,2,3} != uint3{4,2,3});
    REQUIRE(uint3{1,2,3} != uint3{1,4,3});
    REQUIRE(uint3{1,2,3} != uint3{1,2,4});

    REQUIRE(uint4{1,2,3,4} == uint4{1,2,3,4});
    REQUIRE(uint4{1,2,3,4} != uint4{5,2,3,4});
    REQUIRE(uint4{1,2,3,4} != uint4{1,5,3,4});
    REQUIRE(uint4{1,2,3,4} != uint4{1,2,5,4});
    REQUIRE(uint4{1,2,3,4} != uint4{1,2,3,5});
}

TEST_CASE("math_float3_norm", "[math]")
{
    float3 v{1,2,3};
    REQUIRE(stk::norm(v) == Approx(std::sqrt(1.0f*1.0f + 2.0f*2.0f + 3.0f*3.0f)));
    REQUIRE(stk::norm2(v) == Approx(1.0f*1.0f + 2.0f*2.0f + 3.0f*3.0f));
}

TEST_CASE("math_float4_norm", "[math]")
{
    float4 v{1,2,3,4};
    REQUIRE(stk::norm(v) == Approx(std::sqrt(1.0f*1.0f + 2.0f*2.0f + 3.0f*3.0f + 4.0f*4.0f)));
    REQUIRE(stk::norm2(v) == Approx(1.0f*1.0f + 2.0f*2.0f + 3.0f*3.0f + 4.0f*4.0f));
}

TEST_CASE("math_float3_op", "[math]")
{
    float3 v1{1.1f, 2.2f, 3.3f};
    float3 v2{4.4f, 5.5f, 6.6f};

    float3 r1 = v1 + v2;
    REQUIRE(r1.x == Approx(1.1f + 4.4f));
    REQUIRE(r1.y == Approx(2.2f + 5.5f));
    REQUIRE(r1.z == Approx(3.3f + 6.6f));

    float3 r2 = v1 - v2;
    REQUIRE(r2.x == Approx(1.1f - 4.4f));
    REQUIRE(r2.y == Approx(2.2f - 5.5f));
    REQUIRE(r2.z == Approx(3.3f - 6.6f));

    // Element-wise multiplication
    float3 r3 = v1 * v2;
    REQUIRE(r3.x == Approx(1.1f * 4.4f));
    REQUIRE(r3.y == Approx(2.2f * 5.5f));
    REQUIRE(r3.z == Approx(3.3f * 6.6f));

    // Element-wise division
    float3 r4 = v1 / v2;
    REQUIRE(r4.x == Approx(1.1f / 4.4f));
    REQUIRE(r4.y == Approx(2.2f / 5.5f));
    REQUIRE(r4.z == Approx(3.3f / 6.6f));

    float3 r5 = 7.7f * v1;
    REQUIRE(r5.x == Approx(7.7f * 1.1f));
    REQUIRE(r5.y == Approx(7.7f * 2.2f));
    REQUIRE(r5.z == Approx(7.7f * 3.3f));

    float3 r6 = v2 / 8.8f;
    REQUIRE(r6.x == Approx(4.4f / 8.8f));
    REQUIRE(r6.y == Approx(5.5f / 8.8f));
    REQUIRE(r6.z == Approx(6.6f / 8.8f));

}
TEST_CASE("math_float4_op", "[math]")
{
    float4 v1{1.1f, 2.2f, 3.3f, 4.4f};
    float4 v2{5.5f, 6.6f, 7.7f, 8.8f};

    float4 r1 = v1 + v2;
    REQUIRE(r1.x == Approx(1.1f + 5.5f));
    REQUIRE(r1.y == Approx(2.2f + 6.6f));
    REQUIRE(r1.z == Approx(3.3f + 7.7f));
    REQUIRE(r1.w == Approx(4.4f + 8.8f));

    float4 r2 = v1 - v2;
    REQUIRE(r2.x == Approx(1.1f - 5.5f));
    REQUIRE(r2.y == Approx(2.2f - 6.6f));
    REQUIRE(r2.z == Approx(3.3f - 7.7f));
    REQUIRE(r2.w == Approx(4.4f - 8.8f));

    // Element-wise multiplication
    float4 r3 = v1 * v2;
    REQUIRE(r3.x == Approx(1.1f * 5.5f));
    REQUIRE(r3.y == Approx(2.2f * 6.6f));
    REQUIRE(r3.z == Approx(3.3f * 7.7f));
    REQUIRE(r3.w == Approx(4.4f * 8.8f));

    // Element-wise division
    float4 r4 = v1 / v2;
    REQUIRE(r4.x == Approx(1.1f / 5.5f));
    REQUIRE(r4.y == Approx(2.2f / 6.6f));
    REQUIRE(r4.z == Approx(3.3f / 7.7f));
    REQUIRE(r4.w == Approx(4.4f / 8.8f));

    float4 r5 = 9.9f * v1;
    REQUIRE(r5.x == Approx(9.9f * 1.1f));
    REQUIRE(r5.y == Approx(9.9f * 2.2f));
    REQUIRE(r5.z == Approx(9.9f * 3.3f));
    REQUIRE(r5.w == Approx(9.9f * 4.4f));

    float4 r6 = v2 / 10.10f;
    REQUIRE(r6.x == Approx(5.5f / 10.10f));
    REQUIRE(r6.y == Approx(6.6f / 10.10f));
    REQUIRE(r6.z == Approx(7.7f / 10.10f));
    REQUIRE(r6.w == Approx(8.8f / 10.10f));

}
TEST_CASE("math_int3_op", "[math]")
{
    int3 v1{1, 2, 3};
    int3 v2{4, 5, 6};

    // Element-wise multiplication
    int3 r1 = v1 * v2;
    REQUIRE(r1.x == 1 * 4);
    REQUIRE(r1.y == 2 * 5);
    REQUIRE(r1.z == 3 * 6);

    // Element-wise division
    int3 r2 = v1 / v2;
    REQUIRE(r2.x == 1 / 4);
    REQUIRE(r2.y == 2 / 5);
    REQUIRE(r2.z == 3 / 6);

    int3 r3 = v1 + v2;
    REQUIRE(r3.x == 1 + 4);
    REQUIRE(r3.y == 2 + 5);
    REQUIRE(r3.z == 3 + 6);

    int3 r4 = v1 - v2;
    REQUIRE(r4.x == 1 - 4);
    REQUIRE(r4.y == 2 - 5);
    REQUIRE(r4.z == 3 - 6);

    int3 r5 = 10 * v1;
    REQUIRE(r5.x == 10*1);
    REQUIRE(r5.y == 10*2);
    REQUIRE(r5.z == 10*3);

    int3 r6 = v2 / 2;
    REQUIRE(r6.x == 4/2);
    REQUIRE(r6.y == 5/2);
    REQUIRE(r6.z == 6/2);
}
TEST_CASE("math_to_string", "[math]")
{
    SECTION("char2")
    {
        std::ostringstream s;
        s << char2{1, 2};
        REQUIRE(s.str() == "(1 2)");
    }
    SECTION("char3")
    {
        std::ostringstream s;
        s << char3{1, 2, 3};
        REQUIRE(s.str() == "(1 2 3)");
    }
    SECTION("char4")
    {
        std::ostringstream s;
        s << char4{1, 2, 3, 4};
        REQUIRE(s.str() == "(1 2 3 4)");
    }

    SECTION("uchar2")
    {
        std::ostringstream s;
        s << uchar2{1, 2};
        REQUIRE(s.str() == "(1 2)");
    }
    SECTION("uchar3")
    {
        std::ostringstream s;
        s << uchar3{1, 2, 3};
        REQUIRE(s.str() == "(1 2 3)");
    }
    SECTION("uchar4")
    {
        std::ostringstream s;
        s << uchar4{1, 2, 3, 4};
        REQUIRE(s.str() == "(1 2 3 4)");
    }

    SECTION("int2")
    {
        std::ostringstream s;
        s << int2{1, 2};
        REQUIRE(s.str() == "(1 2)");
    }
    SECTION("int3")
    {
        std::ostringstream s;
        s << int3{1, 2, 3};
        REQUIRE(s.str() == "(1 2 3)");
    }
    SECTION("int4")
    {
        std::ostringstream s;
        s << int4{1, 2, 3, 4};
        REQUIRE(s.str() == "(1 2 3 4)");
    }

    SECTION("float2")
    {
        std::ostringstream s;
        s << float2{1.5f, 2.5f};
        REQUIRE(s.str() == "(1.5 2.5)");
    }
    SECTION("float3")
    {
        std::ostringstream s;
        s << float3{1.5f, 2.5f, 3.5f};
        REQUIRE(s.str() == "(1.5 2.5 3.5)");
    }
    SECTION("float4")
    {
        std::ostringstream s;
        s << float4{1.5f, 2.5f, 3.5f, 4.5f};
        REQUIRE(s.str() == "(1.5 2.5 3.5 4.5)");
    }

    SECTION("double2")
    {
        std::ostringstream s;
        s << double2{1.5f, 2.5f};
        REQUIRE(s.str() == "(1.5 2.5)");
    }
    SECTION("double3")
    {
        std::ostringstream s;
        s << double3{1.5f, 2.5f, 3.5f};
        REQUIRE(s.str() == "(1.5 2.5 3.5)");
    }
    SECTION("double4")
    {
        std::ostringstream s;
        s << double4{1.5f, 2.5f, 3.5f, 4.5f};
        REQUIRE(s.str() == "(1.5 2.5 3.5 4.5)");
    }
}


TEST_CASE("math_to_vector", "[math]")
{
    #define COMMA ,
    #define TEST_TYPE(type, btype, n, vals) \
            SECTION(#type) \
            { \
                std::vector<btype> v = stk::to_vector(type(vals)); \
                CHECK(v == std::vector<btype>(vals)); \
            }

    TEST_TYPE(char2, char, 2, {1 COMMA 2});
    TEST_TYPE(char3, char, 3, {1 COMMA 2 COMMA 3});
    TEST_TYPE(char4, char, 4, {1 COMMA 2 COMMA 3 COMMA 4});

    TEST_TYPE(uchar2, unsigned char, 2, {1 COMMA 2});
    TEST_TYPE(uchar3, unsigned char, 3, {1 COMMA 2 COMMA 3});
    TEST_TYPE(uchar4, unsigned char, 4, {1 COMMA 2 COMMA 3 COMMA 4});

    TEST_TYPE(int2, int, 2, {1 COMMA 2});
    TEST_TYPE(int3, int, 3, {1 COMMA 2 COMMA 3});
    TEST_TYPE(int4, int, 4, {1 COMMA 2 COMMA 3 COMMA 4});

    TEST_TYPE(uint2, unsigned int, 2, {1 COMMA 2});
    TEST_TYPE(uint3, unsigned int, 3, {1 COMMA 2 COMMA 3});
    TEST_TYPE(uint4, unsigned int, 4, {1 COMMA 2 COMMA 3 COMMA 4});

    TEST_TYPE(float2, float, 2, {1 COMMA 2});
    TEST_TYPE(float3, float, 3, {1 COMMA 2 COMMA 3});
    TEST_TYPE(float4, float, 4, {1 COMMA 2 COMMA 3 COMMA 4});

    TEST_TYPE(double2, double, 2, {1 COMMA 2});
    TEST_TYPE(double3, double, 3, {1 COMMA 2 COMMA 3});
    TEST_TYPE(double4, double, 4, {1 COMMA 2 COMMA 3 COMMA 4});

    #undef TEST_TYPE
    #undef COMMA

    SECTION("Matrix3x3f")
    {
        Matrix3x3f m = {{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        }};

        std::vector<float> v = stk::to_vector(m);

        CHECK(v == std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8, 9}));
    }
}

TEST_CASE("math_nonzero", "[math]")
{
    CHECK(!stk::nonzero(char2{0, 0}));
    CHECK( stk::nonzero(char2{0, 1}));
    CHECK(!stk::nonzero(char3{0, 0, 0}));
    CHECK( stk::nonzero(char3{0, 1, 0}));
    CHECK(!stk::nonzero(char4{0, 0, 0, 0}));
    CHECK( stk::nonzero(char4{0, 0, 1, 0}));

    CHECK(!stk::nonzero(uchar2{0, 0}));
    CHECK( stk::nonzero(uchar2{0, 1}));
    CHECK(!stk::nonzero(uchar3{0, 0, 0}));
    CHECK( stk::nonzero(uchar3{0, 1, 0}));
    CHECK(!stk::nonzero(uchar4{0, 0, 0, 0}));
    CHECK( stk::nonzero(uchar4{0, 0, 1, 0}));

    CHECK(!stk::nonzero(int2{0, 0}));
    CHECK( stk::nonzero(int2{0, 1}));
    CHECK(!stk::nonzero(int3{0, 0, 0}));
    CHECK( stk::nonzero(int3{0, 1, 0}));
    CHECK(!stk::nonzero(int4{0, 0, 0, 0}));
    CHECK( stk::nonzero(int4{0, 0, 1, 0}));

    CHECK(!stk::nonzero(uint2{0, 0}));
    CHECK( stk::nonzero(uint2{0, 1}));
    CHECK(!stk::nonzero(uint3{0, 0, 0}));
    CHECK( stk::nonzero(uint3{0, 1, 0}));
    CHECK(!stk::nonzero(uint4{0, 0, 0, 0}));
    CHECK( stk::nonzero(uint4{0, 0, 1, 0}));

    const float e = stk::eps() / 2;
    const float f = stk::eps() * 2;

    CHECK(!stk::nonzero(float2{e, e}));
    CHECK( stk::nonzero(float2{f, e}));
    CHECK(!stk::nonzero(float3{e, e, e}));
    CHECK( stk::nonzero(float3{e, f, f}));
    CHECK(!stk::nonzero(float4{e, e, e, e}));
    CHECK( stk::nonzero(float4{e, e, f, e}));

    CHECK(!stk::nonzero(double2{e, e}));
    CHECK( stk::nonzero(double2{f, e}));
    CHECK(!stk::nonzero(double3{e, e, e}));
    CHECK( stk::nonzero(double3{e, f, f}));
    CHECK(!stk::nonzero(double4{e, e, e, e}));
    CHECK( stk::nonzero(double4{e, e, f, e}));
}
