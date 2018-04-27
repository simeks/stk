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

    #define TYPE_ID(T, Id) \
        template<> \
        struct type_id<T> \
        { \
            typedef T Type; \
            enum { \
                id = Id \
            }; \
        };

    TYPE_ID(float, Type_Float);
    TYPE_ID(float2, Type_Float2);
    TYPE_ID(float3, Type_Float3);
    TYPE_ID(float4, Type_Float4);

    TYPE_ID(double, Type_Double);
    TYPE_ID(double2, Type_Double2);
    TYPE_ID(double3, Type_Double3);
    TYPE_ID(double4, Type_Double4);

    TYPE_ID(uint8_t, Type_UChar);
    TYPE_ID(uchar2, Type_UChar2);
    TYPE_ID(uchar3, Type_UChar3);
    TYPE_ID(uchar4, Type_UChar4);

    #undef TYPE_ID
}

