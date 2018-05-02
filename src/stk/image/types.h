#pragma once

#include <stddef.h>
#include <stdint.h>

#include <stk/math/types.h>

namespace stk
{
    enum Type : uint8_t
    {
        Type_Unknown = 0,
        Type_Float,
        Type_Float2,
        Type_Float3,
        Type_Float4,
        Type_Double,
        Type_Double2,
        Type_Double3,
        Type_Double4,
        Type_UChar,
        Type_UChar2,
        Type_UChar3,
        Type_UChar4
    };
    
    // Returns the total size in bytes of the specified type
    size_t type_size(Type type);

    // Returns the number of components of the specified type
    int num_components(Type type);

    // Returns the base type of a type, i.e. Float3 -> Float
    Type base_type(Type type);

    // Returns the name of the type as a string
    const char* as_string(Type type);

    // Type traits
    template<typename T>
    struct type_id
    {
        typedef T Type;
        enum {
            id = Type_Unknown
        };
    };

    // TODO: Annoying thing: you'll have to explicitly typecast type_id<>::id into Type

    #define TYPE_ID(T, BT, Id, NumComp) \
        template<> \
        struct type_id<T> \
        { \
            typedef T Type; \
            typedef BT Base; \
            enum { \
                id = Id, \
                num_comp = NumComp \
            }; \
        };

    TYPE_ID(float, float, Type_Float, 1);
    TYPE_ID(float2, float, Type_Float2, 2);
    TYPE_ID(float3, float, Type_Float3, 3);
    TYPE_ID(float4, float, Type_Float4, 4);

    TYPE_ID(double, double, Type_Double, 1);
    TYPE_ID(double2, double, Type_Double2, 2);
    TYPE_ID(double3, double, Type_Double3, 3);
    TYPE_ID(double4, double, Type_Double4, 4);

    TYPE_ID(uint8_t, uint8_t, Type_UChar, 1);
    TYPE_ID(uchar2, uint8_t, Type_UChar2, 2);
    TYPE_ID(uchar3, uint8_t, Type_UChar3, 3);
    TYPE_ID(uchar4, uint8_t, Type_UChar4, 4);

    #undef TYPE_ID
}

