#include "catch.hpp"

#include <stk/volume/types.h>

using Catch::Matchers::Equals;

TEST_CASE("types", "[volume]")
{
    SECTION("unknown")
    {
        REQUIRE_THAT(stk::as_string(stk::Type_Unknown), Equals("unknown"));
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

        REQUIRE_THAT(stk::as_string(stk::Type_Double), Equals("double"));
        REQUIRE_THAT(stk::as_string(stk::Type_Double2), Equals("double2"));
        REQUIRE_THAT(stk::as_string(stk::Type_Double3), Equals("double3"));
        REQUIRE_THAT(stk::as_string(stk::Type_Double4), Equals("double4"));
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

        REQUIRE_THAT(stk::as_string(stk::Type_UChar), Equals("uchar"));
        REQUIRE_THAT(stk::as_string(stk::Type_UChar2), Equals("uchar2"));
        REQUIRE_THAT(stk::as_string(stk::Type_UChar3), Equals("uchar3"));
        REQUIRE_THAT(stk::as_string(stk::Type_UChar4), Equals("uchar4"));
    }
}
