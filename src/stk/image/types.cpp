#include "types.h"

#include "stk/common/assert.h"

namespace stk
{
    size_t type_size(Type type)
    {
        switch (type)
        {
        case Type_Char:
            return sizeof(char);
        case Type_Char2:
            return sizeof(char) * 2;
        case Type_Char3:
            return sizeof(char) * 3;
        case Type_Char4:
            return sizeof(char) * 4;
        case Type_UChar:
            return sizeof(uint8_t);
        case Type_UChar2:
            return sizeof(uint8_t) * 2;
        case Type_UChar3:
            return sizeof(uint8_t) * 3;
        case Type_UChar4:
            return sizeof(uint8_t) * 4;
        case Type_Short:
            return sizeof(short);
        case Type_Short2:
            return sizeof(short) * 2;
        case Type_Short3:
            return sizeof(short) * 3;
        case Type_Short4:
            return sizeof(short) * 4;
        case Type_UShort:
            return sizeof(uint16_t);
        case Type_UShort2:
            return sizeof(uint16_t) * 2;
        case Type_UShort3:
            return sizeof(uint16_t) * 3;
        case Type_UShort4:
            return sizeof(uint16_t) * 4;
        case Type_Int:
            return sizeof(int);
        case Type_Int2:
            return sizeof(int) * 2;
        case Type_Int3:
            return sizeof(int) * 3;
        case Type_Int4:
            return sizeof(int) * 4;
        case Type_UInt:
            return sizeof(uint32_t);
        case Type_UInt2:
            return sizeof(uint32_t) * 2;
        case Type_UInt3:
            return sizeof(uint32_t) * 3;
        case Type_UInt4:
            return sizeof(uint32_t) * 4;
        case Type_Float:
            return sizeof(float);
        case Type_Float2:
            return sizeof(float) * 2;
        case Type_Float3:
            return sizeof(float) * 3;
        case Type_Float4:
            return sizeof(float) * 4;
        case Type_Double:
            return sizeof(double);
        case Type_Double2:
            return sizeof(double) * 2;
        case Type_Double3:
            return sizeof(double) * 3;
        case Type_Double4:
            return sizeof(double) * 4;
        default:
            DASSERT(false);
        };
        return 0;
    }
    int num_components(Type type)
    {
        switch (type)
        {
        case Type_Char:
        case Type_UChar:
        case Type_Short:
        case Type_UShort:
        case Type_Int:
        case Type_UInt:
        case Type_Float:
        case Type_Double:
            return 1;
        case Type_Char2:
        case Type_UChar2:
        case Type_Short2:
        case Type_UShort2:
        case Type_Int2:
        case Type_UInt2:
        case Type_Float2:
        case Type_Double2:
            return 2;
        case Type_Char3:
        case Type_UChar3:
        case Type_Short3:
        case Type_UShort3:
        case Type_Int3:
        case Type_UInt3:
        case Type_Float3:
        case Type_Double3:
            return 3;
        case Type_Char4:
        case Type_UChar4:
        case Type_Short4:
        case Type_UShort4:
        case Type_Int4:
        case Type_UInt4:
        case Type_Float4:
        case Type_Double4:
            return 4;
        default:
            DASSERT(false);
        };
        return 0;
    }
    Type base_type(Type type)
    {
        switch (type)
        {
        case Type_Char:
        case Type_Char2:
        case Type_Char3:
        case Type_Char4:
            return Type_Char;
        case Type_UChar:
        case Type_UChar2:
        case Type_UChar3:
        case Type_UChar4:
            return Type_UChar;
        case Type_Short:
        case Type_Short2:
        case Type_Short3:
        case Type_Short4:
            return Type_Short;
        case Type_UShort:
        case Type_UShort2:
        case Type_UShort3:
        case Type_UShort4:
            return Type_UShort;
        case Type_Int:
        case Type_Int2:
        case Type_Int3:
        case Type_Int4:
            return Type_Int;
        case Type_UInt:
        case Type_UInt2:
        case Type_UInt3:
        case Type_UInt4:
            return Type_UInt;
        case Type_Float:
        case Type_Float2:
        case Type_Float3:
        case Type_Float4:
            return Type_Float;
        case Type_Double:
        case Type_Double2:
        case Type_Double3:
        case Type_Double4:
            return Type_Double;
        default:
            DASSERT(false);
        };
        return Type_Unknown;
    }
    Type build_type(Type base_type, int num_comp)
    {
        if (base_type == Type_Unknown)
            return Type_Unknown;

        DASSERT(base_type == Type_Char || 
                base_type == Type_UChar ||
                base_type == Type_Short ||
                base_type == Type_UShort ||
                base_type == Type_Int ||
                base_type == Type_UInt ||
                base_type == Type_Float ||
                base_type == Type_Double
        );
        ASSERT(0 < num_comp && num_comp <= 4);
        
        return (Type)((int)base_type + num_comp - 1);
        
    }
    const char* as_string(Type type)
    {
        const char* names[] =
        {
            "unknown",  // Type_Unknown

            "char",     // Type_Char
            "char2",    // Type_Char2
            "char3",    // Type_Char3
            "char4",    // Type_Char4

            "uchar",    // Type_UChar
            "uchar2",   // Type_UChar2
            "uchar3",   // Type_UChar3
            "uchar4",   // Type_UChar4

            "short",    // Type_Short
            "short2",   // Type_Short2
            "short3",   // Type_Short3
            "short4",   // Type_Short4

            "ushort",   // Type_UShort
            "ushort2",  // Type_UShort2
            "ushort3",  // Type_UShort3
            "ushort4",  // Type_UShort4

            "int",    // Type_Int
            "int2",   // Type_Int2
            "int3",   // Type_Int3
            "int4",   // Type_Int4

            "uint",   // Type_UInt
            "uint2",  // Type_UInt2
            "uint3",  // Type_UInt3
            "uint4",  // Type_UInt4

            "float",    // Type_Float
            "float2",   // Type_Float2
            "float3",   // Type_Float3
            "float4",   // Type_Float4

            "double",   // Type_Double
            "double2",  // Type_Double2
            "double3",  // Type_Double3
            "double4",  // Type_Double4
        };
        return names[type];
    }
}
