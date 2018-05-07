#include "catch.hpp"

#include <stk/image/types.h>

using Catch::Matchers::Equals;

TEST_CASE("types", "[volume]")
{
    SECTION("unknown")
    {
        REQUIRE_THAT(stk::as_string(stk::Type_Unknown), Equals("unknown"));
    }
    SECTION("char")
    {
        REQUIRE(stk::type_size(stk::Type_Char) == sizeof(char));
        REQUIRE(stk::type_size(stk::Type_Char2) == sizeof(char2));
        REQUIRE(stk::type_size(stk::Type_Char3) == sizeof(char3));
        REQUIRE(stk::type_size(stk::Type_Char4) == sizeof(char4));

        REQUIRE(stk::num_components(stk::Type_Char) == 1);
        REQUIRE(stk::num_components(stk::Type_Char2) == 2);
        REQUIRE(stk::num_components(stk::Type_Char3) == 3);
        REQUIRE(stk::num_components(stk::Type_Char4) == 4);

        REQUIRE(stk::base_type(stk::Type_Char) == stk::Type_Char);
        REQUIRE(stk::base_type(stk::Type_Char2) == stk::Type_Char);
        REQUIRE(stk::base_type(stk::Type_Char3) == stk::Type_Char);
        REQUIRE(stk::base_type(stk::Type_Char4) == stk::Type_Char);

        REQUIRE(stk::type_id<char>::id == stk::Type_Char);
        REQUIRE(stk::type_id<char2>::id == stk::Type_Char2);
        REQUIRE(stk::type_id<char3>::id == stk::Type_Char3);
        REQUIRE(stk::type_id<char4>::id == stk::Type_Char4);

        REQUIRE(stk::type_id<char>::num_comp == 1);
        REQUIRE(stk::type_id<char2>::num_comp == 2);
        REQUIRE(stk::type_id<char3>::num_comp == 3);
        REQUIRE(stk::type_id<char4>::num_comp == 4);

        REQUIRE_THAT(stk::as_string(stk::Type_Char), Equals("char"));
        REQUIRE_THAT(stk::as_string(stk::Type_Char2), Equals("char2"));
        REQUIRE_THAT(stk::as_string(stk::Type_Char3), Equals("char3"));
        REQUIRE_THAT(stk::as_string(stk::Type_Char4), Equals("char4"));
    }
    SECTION("uchar")
    {
        REQUIRE(stk::type_size(stk::Type_UChar) == sizeof(uint8_t));
        REQUIRE(stk::type_size(stk::Type_UChar2) == sizeof(uchar2));
        REQUIRE(stk::type_size(stk::Type_UChar3) == sizeof(uchar3));
        REQUIRE(stk::type_size(stk::Type_UChar4) == sizeof(uchar4));

        REQUIRE(stk::num_components(stk::Type_UChar) == 1);
        REQUIRE(stk::num_components(stk::Type_UChar2) == 2);
        REQUIRE(stk::num_components(stk::Type_UChar3) == 3);
        REQUIRE(stk::num_components(stk::Type_UChar4) == 4);

        REQUIRE(stk::base_type(stk::Type_UChar) == stk::Type_UChar);
        REQUIRE(stk::base_type(stk::Type_UChar2) == stk::Type_UChar);
        REQUIRE(stk::base_type(stk::Type_UChar3) == stk::Type_UChar);
        REQUIRE(stk::base_type(stk::Type_UChar4) == stk::Type_UChar);

        REQUIRE(stk::type_id<uint8_t>::id == stk::Type_UChar);
        REQUIRE(stk::type_id<uchar2>::id == stk::Type_UChar2);
        REQUIRE(stk::type_id<uchar3>::id == stk::Type_UChar3);
        REQUIRE(stk::type_id<uchar4>::id == stk::Type_UChar4);

        REQUIRE(stk::type_id<uint8_t>::num_comp == 1);
        REQUIRE(stk::type_id<uchar2>::num_comp == 2);
        REQUIRE(stk::type_id<uchar3>::num_comp == 3);
        REQUIRE(stk::type_id<uchar4>::num_comp == 4);

        REQUIRE_THAT(stk::as_string(stk::Type_UChar), Equals("uchar"));
        REQUIRE_THAT(stk::as_string(stk::Type_UChar2), Equals("uchar2"));
        REQUIRE_THAT(stk::as_string(stk::Type_UChar3), Equals("uchar3"));
        REQUIRE_THAT(stk::as_string(stk::Type_UChar4), Equals("uchar4"));
    }
    SECTION("short")
    {
        REQUIRE(stk::type_size(stk::Type_Short) == sizeof(short));
        REQUIRE(stk::type_size(stk::Type_Short2) == sizeof(short2));
        REQUIRE(stk::type_size(stk::Type_Short3) == sizeof(short3));
        REQUIRE(stk::type_size(stk::Type_Short4) == sizeof(short4));

        REQUIRE(stk::num_components(stk::Type_Short) == 1);
        REQUIRE(stk::num_components(stk::Type_Short2) == 2);
        REQUIRE(stk::num_components(stk::Type_Short3) == 3);
        REQUIRE(stk::num_components(stk::Type_Short4) == 4);

        REQUIRE(stk::base_type(stk::Type_Short) == stk::Type_Short);
        REQUIRE(stk::base_type(stk::Type_Short2) == stk::Type_Short);
        REQUIRE(stk::base_type(stk::Type_Short3) == stk::Type_Short);
        REQUIRE(stk::base_type(stk::Type_Short4) == stk::Type_Short);

        REQUIRE(stk::type_id<short>::id == stk::Type_Short);
        REQUIRE(stk::type_id<short2>::id == stk::Type_Short2);
        REQUIRE(stk::type_id<short3>::id == stk::Type_Short3);
        REQUIRE(stk::type_id<short4>::id == stk::Type_Short4);

        REQUIRE(stk::type_id<short>::num_comp == 1);
        REQUIRE(stk::type_id<short2>::num_comp == 2);
        REQUIRE(stk::type_id<short3>::num_comp == 3);
        REQUIRE(stk::type_id<short4>::num_comp == 4);

        REQUIRE_THAT(stk::as_string(stk::Type_Short), Equals("short"));
        REQUIRE_THAT(stk::as_string(stk::Type_Short2), Equals("short2"));
        REQUIRE_THAT(stk::as_string(stk::Type_Short3), Equals("short3"));
        REQUIRE_THAT(stk::as_string(stk::Type_Short4), Equals("short4"));
    }
    SECTION("ushort")
    {
        REQUIRE(stk::type_size(stk::Type_UShort) == sizeof(uint16_t));
        REQUIRE(stk::type_size(stk::Type_UShort2) == sizeof(ushort2));
        REQUIRE(stk::type_size(stk::Type_UShort3) == sizeof(ushort3));
        REQUIRE(stk::type_size(stk::Type_UShort4) == sizeof(ushort4));

        REQUIRE(stk::num_components(stk::Type_UShort) == 1);
        REQUIRE(stk::num_components(stk::Type_UShort2) == 2);
        REQUIRE(stk::num_components(stk::Type_UShort3) == 3);
        REQUIRE(stk::num_components(stk::Type_UShort4) == 4);

        REQUIRE(stk::base_type(stk::Type_UShort) == stk::Type_UShort);
        REQUIRE(stk::base_type(stk::Type_UShort2) == stk::Type_UShort);
        REQUIRE(stk::base_type(stk::Type_UShort3) == stk::Type_UShort);
        REQUIRE(stk::base_type(stk::Type_UShort4) == stk::Type_UShort);

        REQUIRE(stk::type_id<uint16_t>::id == stk::Type_UShort);
        REQUIRE(stk::type_id<ushort2>::id == stk::Type_UShort2);
        REQUIRE(stk::type_id<ushort3>::id == stk::Type_UShort3);
        REQUIRE(stk::type_id<ushort4>::id == stk::Type_UShort4);

        REQUIRE(stk::type_id<uint16_t>::num_comp == 1);
        REQUIRE(stk::type_id<ushort2>::num_comp == 2);
        REQUIRE(stk::type_id<ushort3>::num_comp == 3);
        REQUIRE(stk::type_id<ushort4>::num_comp == 4);

        REQUIRE_THAT(stk::as_string(stk::Type_UShort), Equals("ushort"));
        REQUIRE_THAT(stk::as_string(stk::Type_UShort2), Equals("ushort2"));
        REQUIRE_THAT(stk::as_string(stk::Type_UShort3), Equals("ushort3"));
        REQUIRE_THAT(stk::as_string(stk::Type_UShort4), Equals("ushort4"));
    }
    SECTION("int")
    {
        REQUIRE(stk::type_size(stk::Type_Int) == sizeof(int));
        REQUIRE(stk::type_size(stk::Type_Int2) == sizeof(int2));
        REQUIRE(stk::type_size(stk::Type_Int3) == sizeof(int3));
        REQUIRE(stk::type_size(stk::Type_Int4) == sizeof(int4));

        REQUIRE(stk::num_components(stk::Type_Int) == 1);
        REQUIRE(stk::num_components(stk::Type_Int2) == 2);
        REQUIRE(stk::num_components(stk::Type_Int3) == 3);
        REQUIRE(stk::num_components(stk::Type_Int4) == 4);

        REQUIRE(stk::base_type(stk::Type_Int) == stk::Type_Int);
        REQUIRE(stk::base_type(stk::Type_Int2) == stk::Type_Int);
        REQUIRE(stk::base_type(stk::Type_Int3) == stk::Type_Int);
        REQUIRE(stk::base_type(stk::Type_Int4) == stk::Type_Int);

        REQUIRE(stk::type_id<int>::id == stk::Type_Int);
        REQUIRE(stk::type_id<int2>::id == stk::Type_Int2);
        REQUIRE(stk::type_id<int3>::id == stk::Type_Int3);
        REQUIRE(stk::type_id<int4>::id == stk::Type_Int4);

        REQUIRE(stk::type_id<int>::num_comp == 1);
        REQUIRE(stk::type_id<int2>::num_comp == 2);
        REQUIRE(stk::type_id<int3>::num_comp == 3);
        REQUIRE(stk::type_id<int4>::num_comp == 4);

        REQUIRE_THAT(stk::as_string(stk::Type_Int), Equals("int"));
        REQUIRE_THAT(stk::as_string(stk::Type_Int2), Equals("int2"));
        REQUIRE_THAT(stk::as_string(stk::Type_Int3), Equals("int3"));
        REQUIRE_THAT(stk::as_string(stk::Type_Int4), Equals("int4"));
    }
    SECTION("uint")
    {
        REQUIRE(stk::type_size(stk::Type_UInt) == sizeof(uint32_t));
        REQUIRE(stk::type_size(stk::Type_UInt2) == sizeof(uint2));
        REQUIRE(stk::type_size(stk::Type_UInt3) == sizeof(uint3));
        REQUIRE(stk::type_size(stk::Type_UInt4) == sizeof(uint4));

        REQUIRE(stk::num_components(stk::Type_UInt) == 1);
        REQUIRE(stk::num_components(stk::Type_UInt2) == 2);
        REQUIRE(stk::num_components(stk::Type_UInt3) == 3);
        REQUIRE(stk::num_components(stk::Type_UInt4) == 4);

        REQUIRE(stk::base_type(stk::Type_UInt) == stk::Type_UInt);
        REQUIRE(stk::base_type(stk::Type_UInt2) == stk::Type_UInt);
        REQUIRE(stk::base_type(stk::Type_UInt3) == stk::Type_UInt);
        REQUIRE(stk::base_type(stk::Type_UInt4) == stk::Type_UInt);

        REQUIRE(stk::type_id<uint32_t>::id == stk::Type_UInt);
        REQUIRE(stk::type_id<uint2>::id == stk::Type_UInt2);
        REQUIRE(stk::type_id<uint3>::id == stk::Type_UInt3);
        REQUIRE(stk::type_id<uint4>::id == stk::Type_UInt4);

        REQUIRE(stk::type_id<uint32_t>::num_comp == 1);
        REQUIRE(stk::type_id<uint2>::num_comp == 2);
        REQUIRE(stk::type_id<uint3>::num_comp == 3);
        REQUIRE(stk::type_id<uint4>::num_comp == 4);

        REQUIRE_THAT(stk::as_string(stk::Type_UInt), Equals("uint"));
        REQUIRE_THAT(stk::as_string(stk::Type_UInt2), Equals("uint2"));
        REQUIRE_THAT(stk::as_string(stk::Type_UInt3), Equals("uint3"));
        REQUIRE_THAT(stk::as_string(stk::Type_UInt4), Equals("uint4"));
    }
    SECTION("float")
    {
        REQUIRE(stk::type_size(stk::Type_Float) == sizeof(float));
        REQUIRE(stk::type_size(stk::Type_Float2) == sizeof(float2));
        REQUIRE(stk::type_size(stk::Type_Float3) == sizeof(float3));
        REQUIRE(stk::type_size(stk::Type_Float4) == sizeof(float4));

        REQUIRE(stk::num_components(stk::Type_Float) == 1);
        REQUIRE(stk::num_components(stk::Type_Float2) == 2);
        REQUIRE(stk::num_components(stk::Type_Float3) == 3);
        REQUIRE(stk::num_components(stk::Type_Float4) == 4);

        REQUIRE(stk::base_type(stk::Type_Float) == stk::Type_Float);
        REQUIRE(stk::base_type(stk::Type_Float2) == stk::Type_Float);
        REQUIRE(stk::base_type(stk::Type_Float3) == stk::Type_Float);
        REQUIRE(stk::base_type(stk::Type_Float4) == stk::Type_Float);
            
        REQUIRE(stk::type_id<float>::id == stk::Type_Float);
        REQUIRE(stk::type_id<float2>::id == stk::Type_Float2);
        REQUIRE(stk::type_id<float3>::id == stk::Type_Float3);
        REQUIRE(stk::type_id<float4>::id == stk::Type_Float4);

        REQUIRE(stk::type_id<float>::num_comp == 1);
        REQUIRE(stk::type_id<float2>::num_comp == 2);
        REQUIRE(stk::type_id<float3>::num_comp == 3);
        REQUIRE(stk::type_id<float4>::num_comp == 4);

        REQUIRE_THAT(stk::as_string(stk::Type_Float), Equals("float"));
        REQUIRE_THAT(stk::as_string(stk::Type_Float2), Equals("float2"));
        REQUIRE_THAT(stk::as_string(stk::Type_Float3), Equals("float3"));
        REQUIRE_THAT(stk::as_string(stk::Type_Float4), Equals("float4"));
    }
    SECTION("double")
    {
        REQUIRE(stk::type_size(stk::Type_Double) == sizeof(double));
        REQUIRE(stk::type_size(stk::Type_Double2) == sizeof(double2));
        REQUIRE(stk::type_size(stk::Type_Double3) == sizeof(double3));
        REQUIRE(stk::type_size(stk::Type_Double4) == sizeof(double4));

        REQUIRE(stk::num_components(stk::Type_Double) == 1);
        REQUIRE(stk::num_components(stk::Type_Double2) == 2);
        REQUIRE(stk::num_components(stk::Type_Double3) == 3);
        REQUIRE(stk::num_components(stk::Type_Double4) == 4);

        REQUIRE(stk::base_type(stk::Type_Double) == stk::Type_Double);
        REQUIRE(stk::base_type(stk::Type_Double2) == stk::Type_Double);
        REQUIRE(stk::base_type(stk::Type_Double3) == stk::Type_Double);
        REQUIRE(stk::base_type(stk::Type_Double4) == stk::Type_Double);
            
        REQUIRE(stk::type_id<double>::id == stk::Type_Double);
        REQUIRE(stk::type_id<double2>::id == stk::Type_Double2);
        REQUIRE(stk::type_id<double3>::id == stk::Type_Double3);
        REQUIRE(stk::type_id<double4>::id == stk::Type_Double4);

        REQUIRE(stk::type_id<double>::num_comp == 1);
        REQUIRE(stk::type_id<double2>::num_comp == 2);
        REQUIRE(stk::type_id<double3>::num_comp == 3);
        REQUIRE(stk::type_id<double4>::num_comp == 4);

        REQUIRE_THAT(stk::as_string(stk::Type_Double), Equals("double"));
        REQUIRE_THAT(stk::as_string(stk::Type_Double2), Equals("double2"));
        REQUIRE_THAT(stk::as_string(stk::Type_Double3), Equals("double3"));
        REQUIRE_THAT(stk::as_string(stk::Type_Double4), Equals("double4"));
    }
}
