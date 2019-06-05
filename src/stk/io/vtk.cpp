#include "vtk.h"

#include "stk/common/log.h"
#include "stk/image/volume.h"

#include <assert.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef __GNUC__
    // @bug Missing __builtin_bswap16 in GCC
    #define __byteswap_u16(a) (a<<8)|(a>>8)
    #define __byteswap_u32 __builtin_bswap32
    #define __byteswap_u64 __builtin_bswap64
#else
    #include <intrin.h>

    #define __byteswap_u16 _byteswap_ushort
    #define __byteswap_u32 _byteswap_ulong
    #define __byteswap_u64 _byteswap_uint64
#endif


namespace
{
    // We don't care about version, we only want to know if it's VTK or not
    const char* _vtk_signature = "# vtk";

    void byteswap_16(uint16_t* p, size_t n)
    {
        for (size_t i = 0; i < n; ++i)
            p[i] = __byteswap_u16(p[i]);
    }
    void byteswap_32(uint32_t* p, size_t n)
    {
        for (size_t i = 0; i < n; ++i)
            p[i] = __byteswap_u32(p[i]);
    }
    void byteswap_64(uint64_t* p, size_t n)
    {
        for (size_t i = 0; i < n; ++i)
            p[i] = __byteswap_u64(p[i]);
    }

    void write_big_endian_16(uint16_t* p, size_t n, std::ofstream& f)
    {
        for (size_t i = 0; i < n; ++i)
        {
            uint16_t c = __byteswap_u16(p[i]);
            f.write((const char*)&c, 2);
        }
    }
    void write_big_endian_32(uint32_t* p, size_t n, std::ofstream& f)
    {
        for (size_t i = 0; i < n; ++i)
        {
            uint32_t c = __byteswap_u32(p[i]);
            f.write((const char*)&c, 4);
        }
    }
    void write_big_endian_64(uint64_t* p, size_t n, std::ofstream& f)
    {
        for (size_t i = 0; i < n; ++i)
        {
            uint64_t c = __byteswap_u64(p[i]);
            f.write((const char*)&c, 8);
        }
    }

    stk::Type vtk_string_to_type(const std::string& data_type, int num_comp)
    {
        if (data_type == "char") {
            if (num_comp == 1) return stk::Type_Char;
            if (num_comp == 2) return stk::Type_Char2;
            if (num_comp == 3) return stk::Type_Char3;
            if (num_comp == 4) return stk::Type_Char4;
        }
        else if (data_type == "unsigned_char") {
            if (num_comp == 1) return stk::Type_UChar;
            if (num_comp == 2) return stk::Type_UChar2;
            if (num_comp == 3) return stk::Type_UChar3;
            if (num_comp == 4) return stk::Type_UChar4;
        }
        else if (data_type == "short") {
            if (num_comp == 1) return stk::Type_Short;
            if (num_comp == 2) return stk::Type_Short2;
            if (num_comp == 3) return stk::Type_Short3;
            if (num_comp == 4) return stk::Type_Short4;
        }
        else if (data_type == "unsigned_short") {
            if (num_comp == 1) return stk::Type_UShort;
            if (num_comp == 2) return stk::Type_UShort2;
            if (num_comp == 3) return stk::Type_UShort3;
            if (num_comp == 4) return stk::Type_UShort4;
        }
        else if (data_type == "int") {
            if (num_comp == 1) return stk::Type_Int;
            if (num_comp == 2) return stk::Type_Int2;
            if (num_comp == 3) return stk::Type_Int3;
            if (num_comp == 4) return stk::Type_Int4;
        }
        else if (data_type == "unsigned_int") {
            if (num_comp == 1) return stk::Type_UInt;
            if (num_comp == 2) return stk::Type_UInt2;
            if (num_comp == 3) return stk::Type_UInt3;
            if (num_comp == 4) return stk::Type_UInt4;
        }
        else if (data_type == "double") {
            if (num_comp == 1) return stk::Type_Double;
            if (num_comp == 2) return stk::Type_Double2;
            if (num_comp == 3) return stk::Type_Double3;
            if (num_comp == 4) return stk::Type_Double4;
        }
        else if (data_type == "float") {
            if (num_comp == 1) return stk::Type_Float;
            if (num_comp == 2) return stk::Type_Float2;
            if (num_comp == 3) return stk::Type_Float3;
            if (num_comp == 4) return stk::Type_Float4;
        }
        return stk::Type_Unknown;
    }
}

#undef __byteswap_u16
#undef __byteswap_u32
#undef __byteswap_u64

namespace stk {
namespace vtk {
    Volume read_vtk_volume(const std::string& file, std::stringstream& error)
    {
        // Spec: http://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf

        //# vtk DataFile Version 3.0
        //<Title>
        //BINARY
        //DATASET STRUCTURED_POINTS
        //DIMENSIONS 256 256 740
        //ORIGIN -249.023 -249.023 21.0165
        //SPACING 1.9531 1.9531 2.6001
        //POINT_DATA 48496640
        //SCALARS image_data double
        //LOOKUP_TABLE default

        std::ifstream f(file, std::ios::binary);
        if (!f.is_open()) {
            error << "File not found";
            return Volume();
        }

        std::string line;

        //# vtk DataFile Version 3.0
        std::getline(f, line);
        //<Title>
        std::getline(f, line);

        //BINARY
        std::getline(f, line);
        if (line != "BINARY") {
            error << "Invalid format: " << line;
            f.close();
            return Volume();
        }

        //DATASET STRUCTURED_POINTS
        std::getline(f, line);
        if (line != "DATASET STRUCTURED_POINTS") {
            error << "Unexpected dataset: " << line;
            f.close();
            return Volume();
        }

        // Note: VTK images are axis-aligned, so no direction here
        dim3 size{0, 0, 0};
        float3 origin{0, 0, 0};
        float3 spacing{1, 1, 1};

        // Hopefully we wont have files as large as 2^64
        size_t point_data = size_t(~0);
        Type voxel_type = Type_Unknown;
        int num_comp = 1;

        std::string key;
        std::string value;
        while (std::getline(f, line)) {
            std::stringstream ss(line);

            ss >> key;
            //DIMENSIONS 256 256 740
            if (key == "DIMENSIONS") {
                ss >> value;
                size.x = atoi(value.c_str());
                ss >> value;
                size.y = atoi(value.c_str());
                ss >> value;
                size.z = atoi(value.c_str());
            }
            //ORIGIN - 249.023 - 249.023 21.0165
            else if (key == "ORIGIN") {
                ss >> origin.x;
                ss >> origin.y;
                ss >> origin.z;
            }
            //SPACING 1.9531 1.9531 2.6001
            else if (key == "SPACING") {
                ss >> spacing.x;
                ss >> spacing.y;
                ss >> spacing.z;
            }
            //POINT_DATA 48496640
            else if (key == "POINT_DATA") {
                ss >> value;
                point_data = atoll(value.c_str());
            }
            //SCALARS image_data double
            else if (key == "SCALARS") {
                // SCALARS dataName dataType numComp
                // Here i think numComp is optional

                ss >> value; // dataName, don't know what this is good for
                std::string data_type;
                ss >> data_type;

                if (!ss.eof()) {
                    std::string num_comp_s;
                    ss >> num_comp_s;
                    num_comp = atoi(num_comp_s.c_str());
                }

                voxel_type = vtk_string_to_type(data_type, num_comp);

                if (voxel_type == Type_Unknown) {
                    error << "Unsupported data type: " << data_type << " " << num_comp;
                    f.close();
                    return Volume();
                }
            }
            else if (key == "VECTORS")
            {
                // VECTORS dataName dataType
                // Vectors always have 3 components

                ss >> value; // dataName, don't know what this is good for
                std::string data_type;
                ss >> data_type;

                num_comp = 3;
                voxel_type = vtk_string_to_type(data_type, 3);

                // Assume that blob comes after this line
                break;
            }
            //LOOKUP_TABLE default
            else if (key == "LOOKUP_TABLE") {
                ss >> value;
                if (value != "default") {
                    error << "Invalid parameter to LOOKUP_TABLE: " << value;
                    f.close();
                    return Volume();
                }
                // Assume that blob comes after this line
                break;
            }
        }

        if (size.x == 0 || size.y == 0 || size.z == 0) {
            error << "Invalid volume size: " <<
                size.x << ", " <<
                size.y << ", " <<
                size.z;
            f.close();
            return Volume();
        }

        if (voxel_type == Type_Unknown) {
            error << "Invalid voxel type";
            f.close();
            return Volume();
        }

        if (point_data == size_t(~0)) {
            error << "Invalid point_data";
        }


        // Allocate volume
        Volume vol(size, voxel_type);
        vol.set_origin(origin);
        vol.set_spacing(spacing);

        size_t num_bytes = size.x * size.y * size.z * type_size(voxel_type);
        f.read((char*)vol.ptr(), num_bytes);
        f.close();

        // Switch to little endian
        size_t bytes_per_elem = type_size(voxel_type) / num_comp;
        size_t num_values = size.x * size.y * size.z *  num_comp;
        if (bytes_per_elem == 8) // double, uint64_t
            byteswap_64((uint64_t*)vol.ptr(), num_values);
        if (bytes_per_elem == 4) // float, uint32_t, ...
            byteswap_32((uint32_t*)vol.ptr(), num_values);
        if (bytes_per_elem == 2) // short
            byteswap_16((uint16_t*)vol.ptr(), num_values);

        // We don't need to do anything for 1 byte elements

        return vol;
    }

    Volume read(const std::string& filename)
    {
        std::stringstream ss;
        Volume vol = read_vtk_volume(filename, ss);

        LOG_IF(Error, !vol.valid()) << ss.str();

        return vol;
    }

    size_t signature_length()
    {
        // First line in file
        // "# vtk"
        return 5;
    }

    bool can_read(const std::string& /*filename*/, const char* signature, size_t len)
    {
        return len >= signature_length() &&
                memcmp(signature, _vtk_signature, signature_length()) == 0;
    }

    void write(const std::string& file, const Volume& vol)
    {
        ASSERT(vol.valid());
        ASSERT(vol.voxel_type() != Type_Unknown);

        //# vtk DataFile Version 3.0
        //<Title>
        //BINARY
        //DATASET STRUCTURED_POINTS
        //DIMENSIONS 256 256 740
        //ORIGIN -249.023 -249.023 21.0165
        //SPACING 1.9531 1.9531 2.6001
        //POINT_DATA 48496640
        //SCALARS image_data double
        //LOOKUP_TABLE default

        std::ofstream f(file, std::ios::binary);
        assert(f.is_open());
        f.setf(std::ios::scientific,
                    std::ios::floatfield);

        f.precision(16);

        f << "# vtk DataFile Version 3.0\n";
        f << "Written by cortado (vtk.cpp)\n";
        f << "BINARY\n";
        f << "DATASET STRUCTURED_POINTS\n";

        auto size = vol.size();
        f << "DIMENSIONS " << size.x << " " << size.y << " " << size.z << "\n";
        float3 origin = vol.origin();
        f << "ORIGIN " << origin.x << " " << origin.y << " " << origin.z << "\n";
        float3 spacing = vol.spacing();
        f << "SPACING " << spacing.x << " " << spacing.y << " " << spacing.z << "\n";
        f << "POINT_DATA " << size.x * size.y * size.z << "\n";

        const char* data_type;
        int num_comp = num_components(vol.voxel_type());
        switch (base_type(vol.voxel_type())) {
        case Type_Char:
            data_type = "char";
            break;
        case Type_UChar:
            data_type = "unsigned_char";
            break;
        case Type_Short:
            data_type = "short";
            break;
        case Type_UShort:
            data_type = "unsigned_short";
            break;
        case Type_Int:
            data_type = "int";
            break;
        case Type_UInt:
            data_type = "unsigned_int";
            break;
        case Type_Float:
            data_type = "float";
            break;
        case Type_Double:
            data_type = "double";
            break;
        default:
            FATAL() << "Unsupported format";
            return;
        };

        f << "SCALARS image_data " << data_type << " " << num_comp << "\n";
        f << "LOOKUP_TABLE default\n";

        // Switch to big endian
        size_t bytes_per_elem = type_size(vol.voxel_type()) / num_comp;
        size_t num_values = size.x * size.y * size.z * num_comp;

        if (bytes_per_elem == 8) // double, uint64_t
            write_big_endian_64((uint64_t*)vol.ptr(), num_values, f);
        else if (bytes_per_elem == 4) // float, uint32_t, ...
            write_big_endian_32((uint32_t*)vol.ptr(), num_values, f);
        else if (bytes_per_elem == 2) // short
            write_big_endian_16((uint16_t*)vol.ptr(), num_values, f);
        else
            f.write((const char*)vol.ptr(), num_values);

        f.close();
    }

    bool can_write(const std::string& filename)
    {
        size_t p = filename.rfind('.');
        if (p == std::string::npos)
            return false;

        std::string ext = filename.substr(p+1); // Skip '.'

        for (size_t i = 0; i < ext.length(); ++i)
            ext[i] = (char)::tolower(ext[i]);

        return ext == "vtk";
    }

} // namespace vtk
} // namespace stk
