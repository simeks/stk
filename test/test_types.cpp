#include "catch.hpp"

#include <stk/image/types.h>

using namespace stk;

using Catch::Matchers::Equals;

TEST_CASE("types", "[volume]")
{
    SECTION("unknown")
    {
        REQUIRE_THAT(as_string(Type_Unknown), Equals("unknown"));
    }
    SECTION("char")
    {
        REQUIRE(type_size(Type_Char) == sizeof(char));
        REQUIRE(type_size(Type_Char2) == sizeof(char2));
        REQUIRE(type_size(Type_Char3) == sizeof(char3));
        REQUIRE(type_size(Type_Char4) == sizeof(char4));

        REQUIRE(num_components(Type_Char) == 1);
        REQUIRE(num_components(Type_Char2) == 2);
        REQUIRE(num_components(Type_Char3) == 3);
        REQUIRE(num_components(Type_Char4) == 4);

        REQUIRE(base_type(Type_Char) == Type_Char);
        REQUIRE(base_type(Type_Char2) == Type_Char);
        REQUIRE(base_type(Type_Char3) == Type_Char);
        REQUIRE(base_type(Type_Char4) == Type_Char);

        REQUIRE(type_id<char>::id() == Type_Char);
        REQUIRE(type_id<char2>::id() == Type_Char2);
        REQUIRE(type_id<char3>::id() == Type_Char3);
        REQUIRE(type_id<char4>::id() == Type_Char4);

        REQUIRE(type_id<char>::num_comp() == 1);
        REQUIRE(type_id<char2>::num_comp() == 2);
        REQUIRE(type_id<char3>::num_comp() == 3);
        REQUIRE(type_id<char4>::num_comp() == 4);

        REQUIRE_THAT(as_string(Type_Char), Equals("char"));
        REQUIRE_THAT(as_string(Type_Char2), Equals("char2"));
        REQUIRE_THAT(as_string(Type_Char3), Equals("char3"));
        REQUIRE_THAT(as_string(Type_Char4), Equals("char4"));
    }
    SECTION("uchar")
    {
        REQUIRE(type_size(Type_UChar) == sizeof(uint8_t));
        REQUIRE(type_size(Type_UChar2) == sizeof(uchar2));
        REQUIRE(type_size(Type_UChar3) == sizeof(uchar3));
        REQUIRE(type_size(Type_UChar4) == sizeof(uchar4));

        REQUIRE(num_components(Type_UChar) == 1);
        REQUIRE(num_components(Type_UChar2) == 2);
        REQUIRE(num_components(Type_UChar3) == 3);
        REQUIRE(num_components(Type_UChar4) == 4);

        REQUIRE(base_type(Type_UChar) == Type_UChar);
        REQUIRE(base_type(Type_UChar2) == Type_UChar);
        REQUIRE(base_type(Type_UChar3) == Type_UChar);
        REQUIRE(base_type(Type_UChar4) == Type_UChar);

        REQUIRE(type_id<uint8_t>::id() == Type_UChar);
        REQUIRE(type_id<uchar2>::id() == Type_UChar2);
        REQUIRE(type_id<uchar3>::id() == Type_UChar3);
        REQUIRE(type_id<uchar4>::id() == Type_UChar4);

        REQUIRE(type_id<uint8_t>::num_comp() == 1);
        REQUIRE(type_id<uchar2>::num_comp() == 2);
        REQUIRE(type_id<uchar3>::num_comp() == 3);
        REQUIRE(type_id<uchar4>::num_comp() == 4);

        REQUIRE_THAT(as_string(Type_UChar), Equals("uchar"));
        REQUIRE_THAT(as_string(Type_UChar2), Equals("uchar2"));
        REQUIRE_THAT(as_string(Type_UChar3), Equals("uchar3"));
        REQUIRE_THAT(as_string(Type_UChar4), Equals("uchar4"));
    }
    SECTION("short")
    {
        REQUIRE(type_size(Type_Short) == sizeof(short));
        REQUIRE(type_size(Type_Short2) == sizeof(short2));
        REQUIRE(type_size(Type_Short3) == sizeof(short3));
        REQUIRE(type_size(Type_Short4) == sizeof(short4));

        REQUIRE(num_components(Type_Short) == 1);
        REQUIRE(num_components(Type_Short2) == 2);
        REQUIRE(num_components(Type_Short3) == 3);
        REQUIRE(num_components(Type_Short4) == 4);

        REQUIRE(base_type(Type_Short) == Type_Short);
        REQUIRE(base_type(Type_Short2) == Type_Short);
        REQUIRE(base_type(Type_Short3) == Type_Short);
        REQUIRE(base_type(Type_Short4) == Type_Short);

        REQUIRE(type_id<short>::id() == Type_Short);
        REQUIRE(type_id<short2>::id() == Type_Short2);
        REQUIRE(type_id<short3>::id() == Type_Short3);
        REQUIRE(type_id<short4>::id() == Type_Short4);

        REQUIRE(type_id<short>::num_comp() == 1);
        REQUIRE(type_id<short2>::num_comp() == 2);
        REQUIRE(type_id<short3>::num_comp() == 3);
        REQUIRE(type_id<short4>::num_comp() == 4);

        REQUIRE_THAT(as_string(Type_Short), Equals("short"));
        REQUIRE_THAT(as_string(Type_Short2), Equals("short2"));
        REQUIRE_THAT(as_string(Type_Short3), Equals("short3"));
        REQUIRE_THAT(as_string(Type_Short4), Equals("short4"));
    }
    SECTION("ushort")
    {
        REQUIRE(type_size(Type_UShort) == sizeof(uint16_t));
        REQUIRE(type_size(Type_UShort2) == sizeof(ushort2));
        REQUIRE(type_size(Type_UShort3) == sizeof(ushort3));
        REQUIRE(type_size(Type_UShort4) == sizeof(ushort4));

        REQUIRE(num_components(Type_UShort) == 1);
        REQUIRE(num_components(Type_UShort2) == 2);
        REQUIRE(num_components(Type_UShort3) == 3);
        REQUIRE(num_components(Type_UShort4) == 4);

        REQUIRE(base_type(Type_UShort) == Type_UShort);
        REQUIRE(base_type(Type_UShort2) == Type_UShort);
        REQUIRE(base_type(Type_UShort3) == Type_UShort);
        REQUIRE(base_type(Type_UShort4) == Type_UShort);

        REQUIRE(type_id<uint16_t>::id() == Type_UShort);
        REQUIRE(type_id<ushort2>::id() == Type_UShort2);
        REQUIRE(type_id<ushort3>::id() == Type_UShort3);
        REQUIRE(type_id<ushort4>::id() == Type_UShort4);

        REQUIRE(type_id<uint16_t>::num_comp() == 1);
        REQUIRE(type_id<ushort2>::num_comp() == 2);
        REQUIRE(type_id<ushort3>::num_comp() == 3);
        REQUIRE(type_id<ushort4>::num_comp() == 4);

        REQUIRE_THAT(as_string(Type_UShort), Equals("ushort"));
        REQUIRE_THAT(as_string(Type_UShort2), Equals("ushort2"));
        REQUIRE_THAT(as_string(Type_UShort3), Equals("ushort3"));
        REQUIRE_THAT(as_string(Type_UShort4), Equals("ushort4"));
    }
    SECTION("int")
    {
        REQUIRE(type_size(Type_Int) == sizeof(int));
        REQUIRE(type_size(Type_Int2) == sizeof(int2));
        REQUIRE(type_size(Type_Int3) == sizeof(int3));
        REQUIRE(type_size(Type_Int4) == sizeof(int4));

        REQUIRE(num_components(Type_Int) == 1);
        REQUIRE(num_components(Type_Int2) == 2);
        REQUIRE(num_components(Type_Int3) == 3);
        REQUIRE(num_components(Type_Int4) == 4);

        REQUIRE(base_type(Type_Int) == Type_Int);
        REQUIRE(base_type(Type_Int2) == Type_Int);
        REQUIRE(base_type(Type_Int3) == Type_Int);
        REQUIRE(base_type(Type_Int4) == Type_Int);

        REQUIRE(type_id<int>::id() == Type_Int);
        REQUIRE(type_id<int2>::id() == Type_Int2);
        REQUIRE(type_id<int3>::id() == Type_Int3);
        REQUIRE(type_id<int4>::id() == Type_Int4);

        REQUIRE(type_id<int>::num_comp() == 1);
        REQUIRE(type_id<int2>::num_comp() == 2);
        REQUIRE(type_id<int3>::num_comp() == 3);
        REQUIRE(type_id<int4>::num_comp() == 4);

        REQUIRE_THAT(as_string(Type_Int), Equals("int"));
        REQUIRE_THAT(as_string(Type_Int2), Equals("int2"));
        REQUIRE_THAT(as_string(Type_Int3), Equals("int3"));
        REQUIRE_THAT(as_string(Type_Int4), Equals("int4"));
    }
    SECTION("uint")
    {
        REQUIRE(type_size(Type_UInt) == sizeof(uint32_t));
        REQUIRE(type_size(Type_UInt2) == sizeof(uint2));
        REQUIRE(type_size(Type_UInt3) == sizeof(uint3));
        REQUIRE(type_size(Type_UInt4) == sizeof(uint4));

        REQUIRE(num_components(Type_UInt) == 1);
        REQUIRE(num_components(Type_UInt2) == 2);
        REQUIRE(num_components(Type_UInt3) == 3);
        REQUIRE(num_components(Type_UInt4) == 4);

        REQUIRE(base_type(Type_UInt) == Type_UInt);
        REQUIRE(base_type(Type_UInt2) == Type_UInt);
        REQUIRE(base_type(Type_UInt3) == Type_UInt);
        REQUIRE(base_type(Type_UInt4) == Type_UInt);

        REQUIRE(type_id<uint32_t>::id() == Type_UInt);
        REQUIRE(type_id<uint2>::id() == Type_UInt2);
        REQUIRE(type_id<uint3>::id() == Type_UInt3);
        REQUIRE(type_id<uint4>::id() == Type_UInt4);

        REQUIRE(type_id<uint32_t>::num_comp() == 1);
        REQUIRE(type_id<uint2>::num_comp() == 2);
        REQUIRE(type_id<uint3>::num_comp() == 3);
        REQUIRE(type_id<uint4>::num_comp() == 4);

        REQUIRE_THAT(as_string(Type_UInt), Equals("uint"));
        REQUIRE_THAT(as_string(Type_UInt2), Equals("uint2"));
        REQUIRE_THAT(as_string(Type_UInt3), Equals("uint3"));
        REQUIRE_THAT(as_string(Type_UInt4), Equals("uint4"));
    }
    SECTION("float")
    {
        REQUIRE(type_size(Type_Float) == sizeof(float));
        REQUIRE(type_size(Type_Float2) == sizeof(float2));
        REQUIRE(type_size(Type_Float3) == sizeof(float3));
        REQUIRE(type_size(Type_Float4) == sizeof(float4));

        REQUIRE(num_components(Type_Float) == 1);
        REQUIRE(num_components(Type_Float2) == 2);
        REQUIRE(num_components(Type_Float3) == 3);
        REQUIRE(num_components(Type_Float4) == 4);

        REQUIRE(base_type(Type_Float) == Type_Float);
        REQUIRE(base_type(Type_Float2) == Type_Float);
        REQUIRE(base_type(Type_Float3) == Type_Float);
        REQUIRE(base_type(Type_Float4) == Type_Float);
            
        REQUIRE(type_id<float>::id() == Type_Float);
        REQUIRE(type_id<float2>::id() == Type_Float2);
        REQUIRE(type_id<float3>::id() == Type_Float3);
        REQUIRE(type_id<float4>::id() == Type_Float4);

        REQUIRE(type_id<float>::num_comp() == 1);
        REQUIRE(type_id<float2>::num_comp() == 2);
        REQUIRE(type_id<float3>::num_comp() == 3);
        REQUIRE(type_id<float4>::num_comp() == 4);

        REQUIRE_THAT(as_string(Type_Float), Equals("float"));
        REQUIRE_THAT(as_string(Type_Float2), Equals("float2"));
        REQUIRE_THAT(as_string(Type_Float3), Equals("float3"));
        REQUIRE_THAT(as_string(Type_Float4), Equals("float4"));
    }
    SECTION("double")
    {
        REQUIRE(type_size(Type_Double) == sizeof(double));
        REQUIRE(type_size(Type_Double2) == sizeof(double2));
        REQUIRE(type_size(Type_Double3) == sizeof(double3));
        REQUIRE(type_size(Type_Double4) == sizeof(double4));

        REQUIRE(num_components(Type_Double) == 1);
        REQUIRE(num_components(Type_Double2) == 2);
        REQUIRE(num_components(Type_Double3) == 3);
        REQUIRE(num_components(Type_Double4) == 4);

        REQUIRE(base_type(Type_Double) == Type_Double);
        REQUIRE(base_type(Type_Double2) == Type_Double);
        REQUIRE(base_type(Type_Double3) == Type_Double);
        REQUIRE(base_type(Type_Double4) == Type_Double);
            
        REQUIRE(type_id<double>::id() == Type_Double);
        REQUIRE(type_id<double2>::id() == Type_Double2);
        REQUIRE(type_id<double3>::id() == Type_Double3);
        REQUIRE(type_id<double4>::id() == Type_Double4);

        REQUIRE(type_id<double>::num_comp() == 1);
        REQUIRE(type_id<double2>::num_comp() == 2);
        REQUIRE(type_id<double3>::num_comp() == 3);
        REQUIRE(type_id<double4>::num_comp() == 4);

        REQUIRE_THAT(as_string(Type_Double), Equals("double"));
        REQUIRE_THAT(as_string(Type_Double2), Equals("double2"));
        REQUIRE_THAT(as_string(Type_Double3), Equals("double3"));
        REQUIRE_THAT(as_string(Type_Double4), Equals("double4"));
    }
}
TEST_CASE("build_type", "[volume]")
{
    REQUIRE(build_type(Type_Char, 1) == Type_Char);
    REQUIRE(build_type(Type_Char, 2) == Type_Char2);
    REQUIRE(build_type(Type_Char, 3) == Type_Char3);
    REQUIRE(build_type(Type_Char, 4) == Type_Char4);

    REQUIRE(build_type(Type_UChar, 1) == Type_UChar);
    REQUIRE(build_type(Type_UChar, 2) == Type_UChar2);
    REQUIRE(build_type(Type_UChar, 3) == Type_UChar3);
    REQUIRE(build_type(Type_UChar, 4) == Type_UChar4);

    REQUIRE(build_type(Type_Short, 1) == Type_Short);
    REQUIRE(build_type(Type_Short, 2) == Type_Short2);
    REQUIRE(build_type(Type_Short, 3) == Type_Short3);
    REQUIRE(build_type(Type_Short, 4) == Type_Short4);

    REQUIRE(build_type(Type_UShort, 1) == Type_UShort);
    REQUIRE(build_type(Type_UShort, 2) == Type_UShort2);
    REQUIRE(build_type(Type_UShort, 3) == Type_UShort3);
    REQUIRE(build_type(Type_UShort, 4) == Type_UShort4);

    REQUIRE(build_type(Type_Int, 1) == Type_Int);
    REQUIRE(build_type(Type_Int, 2) == Type_Int2);
    REQUIRE(build_type(Type_Int, 3) == Type_Int3);
    REQUIRE(build_type(Type_Int, 4) == Type_Int4);

    REQUIRE(build_type(Type_UInt, 1) == Type_UInt);
    REQUIRE(build_type(Type_UInt, 2) == Type_UInt2);
    REQUIRE(build_type(Type_UInt, 3) == Type_UInt3);
    REQUIRE(build_type(Type_UInt, 4) == Type_UInt4);

    REQUIRE(build_type(Type_Float, 1) == Type_Float);
    REQUIRE(build_type(Type_Float, 2) == Type_Float2);
    REQUIRE(build_type(Type_Float, 3) == Type_Float3);
    REQUIRE(build_type(Type_Float, 4) == Type_Float4);

    REQUIRE(build_type(Type_Double, 1) == Type_Double);
    REQUIRE(build_type(Type_Double, 2) == Type_Double2);
    REQUIRE(build_type(Type_Double, 3) == Type_Double3);
    REQUIRE(build_type(Type_Double, 4) == Type_Double4);
}
