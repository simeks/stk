#pragma once

#include <stddef.h>
#include <stdint.h>

#include <stk/math/types.h>

namespace stk
{
    enum Type : uint8_t
    {
        // Do not change the orders of these entries as some functions are
        //  dependent on this. E.g. build_type

        Type_Unknown = 0,

        Type_Char,
        Type_Char2,
        Type_Char3,
        Type_Char4,

        Type_UChar,
        Type_UChar2,
        Type_UChar3,
        Type_UChar4,

        Type_Short,
        Type_Short2,
        Type_Short3,
        Type_Short4,

        Type_UShort,
        Type_UShort2,
        Type_UShort3,
        Type_UShort4,

        Type_Int,
        Type_Int2,
        Type_Int3,
        Type_Int4,

        Type_UInt,
        Type_UInt2,
        Type_UInt3,
        Type_UInt4,

        Type_Float,
        Type_Float2,
        Type_Float3,
        Type_Float4,

        Type_Double,
        Type_Double2,
        Type_Double3,
        Type_Double4
    };

    // Returns the total size in bytes of the specified type
    size_t type_size(Type type);

    // Returns the number of components of the specified type
    int num_components(Type type);

    // Returns the base type of a type, i.e. Float3 -> Float
    Type base_type(Type type);

    // Returns the name of the type as a string
    const char* as_string(Type type);

    // Combines a base type and number of components and returns the new type.
    // base_type has to be a base type, e.g. Type_Float.
    Type build_type(Type base_type, int num_comp);

    // Type traits
    template<typename T>
    struct type_id
    {
        typedef T Type;
        static constexpr stk::Type id(void) {return Type_Unknown;};
    };

    #define TYPE_ID(T, BT, Id, NumComp) \
        template<> \
        struct type_id<T> \
        { \
            typedef T Type; \
            typedef BT Base; \
            static constexpr stk::Type id() {return Id;}; \
            static constexpr int num_comp() {return NumComp;}; \
        }

    TYPE_ID(char, char, Type_Char, 1);
    TYPE_ID(char2, char, Type_Char2, 2);
    TYPE_ID(char3, char, Type_Char3, 3);
    TYPE_ID(char4, char, Type_Char4, 4);

    TYPE_ID(uint8_t, uint8_t, Type_UChar, 1);
    TYPE_ID(uchar2, uint8_t, Type_UChar2, 2);
    TYPE_ID(uchar3, uint8_t, Type_UChar3, 3);
    TYPE_ID(uchar4, uint8_t, Type_UChar4, 4);

    TYPE_ID(short, short, Type_Short, 1);
    TYPE_ID(short2, short, Type_Short2, 2);
    TYPE_ID(short3, short, Type_Short3, 3);
    TYPE_ID(short4, short, Type_Short4, 4);

    TYPE_ID(uint16_t, uint16_t, Type_UShort, 1);
    TYPE_ID(ushort2, uint16_t, Type_UShort2, 2);
    TYPE_ID(ushort3, uint16_t, Type_UShort3, 3);
    TYPE_ID(ushort4, uint16_t, Type_UShort4, 4);

    TYPE_ID(int, int, Type_Int, 1);
    TYPE_ID(int2, int, Type_Int2, 2);
    TYPE_ID(int3, int, Type_Int3, 3);
    TYPE_ID(int4, int, Type_Int4, 4);

    TYPE_ID(uint32_t, uint32_t, Type_UInt, 1);
    TYPE_ID(uint2, uint32_t, Type_UInt2, 2);
    TYPE_ID(uint3, uint32_t, Type_UInt3, 3);
    TYPE_ID(uint4, uint32_t, Type_UInt4, 4);

    TYPE_ID(float, float, Type_Float, 1);
    TYPE_ID(float2, float, Type_Float2, 2);
    TYPE_ID(float3, float, Type_Float3, 3);
    TYPE_ID(float4, float, Type_Float4, 4);

    TYPE_ID(double, double, Type_Double, 1);
    TYPE_ID(double2, double, Type_Double2, 2);
    TYPE_ID(double3, double, Type_Double3, 3);
    TYPE_ID(double4, double, Type_Double4, 4);

    #undef TYPE_ID
}
