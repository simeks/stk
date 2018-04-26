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

    // Type traits
    template<typename T>
    struct type_id
    {
        typedef T Type;
        enum {
            id = Type_Unknown
        };
    };

    #define TYPE_TRAIT(T, Id) \
        template<> \
        struct type_id<T> \
        { \
            typedef T Type; \
            enum { \
                id = Id \
            }; \
        };

    TYPE_TRAIT(float, Type_Float);
    TYPE_TRAIT(float2, Type_Float2);
    TYPE_TRAIT(float3, Type_Float3);
    TYPE_TRAIT(float4, Type_Float4);

    TYPE_TRAIT(double, Type_Double);
    TYPE_TRAIT(double2, Type_Double2);
    TYPE_TRAIT(double3, Type_Double3);
    TYPE_TRAIT(double4, Type_Double4);

    TYPE_TRAIT(uint8_t, Type_UChar);
    TYPE_TRAIT(uchar2, Type_UChar2);
    TYPE_TRAIT(uchar3, Type_UChar3);
    TYPE_TRAIT(uchar4, Type_UChar4);

    #undef TYPE_TRAIT
}
