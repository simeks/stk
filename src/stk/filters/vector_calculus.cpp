#include "vector_calculus.h"
#include "map.h"

namespace {

template<typename TVoxelType, typename TOutputType = float3>
stk::VolumeHelper<TOutputType> nabla(
        const stk::VolumeHelper<TVoxelType>& def
        )
{
    stk::VolumeHelper<TOutputType> out(def.size());
    out.copy_meta_from(def);

    const float3 s = float3{0.5, 0.5, 0.5} / def.spacing();
    const stk::BorderMode bm = stk::Border_Replicate;
    using T = typename stk::type_id<TOutputType>::Base;

    #pragma omp parallel for
    for (int z = 0; z < (int) def.size().z; ++z) {
        for (int y = 0; y < (int) def.size().y; ++y) {
            for (int x = 0; x < (int) def.size().x; ++x) {
                out(x, y, z) = TOutputType{
                    s.x * T(def.at(x+1, y,   z,   bm) - def.at(x-1, y,   z,   bm)),
                    s.y * T(def.at(x,   y+1, z,   bm) - def.at(x,   y-1, z,   bm)),
                    s.z * T(def.at(x,   y,   z+1, bm) - def.at(x,   y,   z-1, bm))
                };
            }
        }
    }

    return out;
}

template<typename TVoxelType, typename TOutputType = float>
stk::VolumeHelper<TOutputType> divergence(
        const stk::VolumeHelper<TVoxelType>& def
        )
{
    stk::VolumeHelper<TOutputType> out(def.size());
    out.copy_meta_from(def);

    const float3 s = float3{0.5, 0.5, 0.5} / def.spacing();
    const stk::BorderMode bm = stk::Border_Replicate;
    using T = TOutputType;

    #pragma omp parallel for
    for (int z = 0; z < (int) def.size().z; ++z) {
        for (int y = 0; y < (int) def.size().y; ++y) {
            for (int x = 0; x < (int) def.size().x; ++x) {
                out(x, y, z) =
                    s.x * (T(def.at(x+1, y,   z,   bm).x) - T(def.at(x-1, y,   z,   bm).x)) +
                    s.y * (T(def.at(x,   y+1, z,   bm).y) - T(def.at(x,   y-1, z,   bm).y)) +
                    s.z * (T(def.at(x,   y,   z+1, bm).z) - T(def.at(x,   y,   z-1, bm).z));
            }
        }
    }

    return out;
}

template<typename TVoxelType, typename TOutputType = float3>
stk::VolumeHelper<TOutputType> rotor(
        const stk::VolumeHelper<TVoxelType>& def
        )
{
    stk::VolumeHelper<TOutputType> out(def.size());
    out.copy_meta_from(def);

    const float3 s = float3{0.5, 0.5, 0.5} / def.spacing();
    const stk::BorderMode bm = stk::Border_Replicate;
    using T = typename stk::type_id<TOutputType>::Base;

    #pragma omp parallel for
    for (int z = 0; z < (int) def.size().z; ++z) {
        for (int y = 0; y < (int) def.size().y; ++y) {
            for (int x = 0; x < (int) def.size().x; ++x) {
                T dx_dy = s.y * T(def.at(x,   y+1, z,   bm).x - def.at(x,   y-1, z,   bm).x);
                T dx_dz = s.z * T(def.at(x,   y,   z+1, bm).x - def.at(x,   y,   z-1, bm).x);
                T dy_dx = s.x * T(def.at(x+1, y,   z,   bm).y - def.at(x-1, y,   z,   bm).y);
                T dy_dz = s.z * T(def.at(x,   y,   z+1, bm).y - def.at(x,   y,   z-1, bm).y);
                T dz_dx = s.x * T(def.at(x+1, y,   z,   bm).z - def.at(x-1, y,   z,   bm).z);
                T dz_dy = s.y * T(def.at(x,   y+1, z,   bm).z - def.at(x,   y-1, z,   bm).z);
                out(x, y, z) = TOutputType{dz_dy - dy_dz, dx_dz - dz_dx, dy_dx - dx_dy};
            }
        }
    }

    return out;
}

} // namespace

namespace stk {

Volume nabla(const Volume& displacement)
{
    switch (displacement.voxel_type())
    {
    case Type_UChar:
        return ::nabla<unsigned char>(displacement);
    case Type_Char:
        return ::nabla<char>(displacement);
    case Type_UShort:
        return ::nabla<unsigned short>(displacement);
    case Type_Short:
        return ::nabla<short>(displacement);
    case Type_UInt:
        return ::nabla<unsigned int>(displacement);
    case Type_Int:
        return ::nabla<int>(displacement);
    case Type_Float:
        return ::nabla<float>(displacement);
    case Type_Double:
        return ::nabla<double>(displacement);
    default:
        break;
    };
    FATAL() << "Unsupported voxel type "
            << "'" << stk::as_string(displacement.voxel_type()) << "'";
}

Volume divergence(const Volume& displacement)
{
    switch (displacement.voxel_type())
    {
    case Type_UChar3:
        return ::divergence<uchar3>(displacement);
    case Type_Char3:
        return ::divergence<char3>(displacement);
    case Type_UShort3:
        return ::divergence<ushort3>(displacement);
    case Type_Short3:
        return ::divergence<short3>(displacement);
    case Type_UInt3:
        return ::divergence<uint3>(displacement);
    case Type_Int3:
        return ::divergence<int3>(displacement);
    case Type_Float3:
        return ::divergence<float3>(displacement);
    case Type_Double3:
        return ::divergence<double3>(displacement);
    default:
        break;
    };
    FATAL() << "Unsupported voxel type "
            << "'" << stk::as_string(displacement.voxel_type()) << "'";
}

Volume rotor(const Volume& displacement)
{
    switch (displacement.voxel_type())
    {
    case Type_UChar3:
        return ::rotor<uchar3>(displacement);
    case Type_Char3:
        return ::rotor<char3>(displacement);
    case Type_UShort:
        return ::rotor<ushort3>(displacement);
    case Type_Short:
        return ::rotor<short3>(displacement);
    case Type_UInt3:
        return ::rotor<uint3>(displacement);
    case Type_Int3:
        return ::rotor<int3>(displacement);
    case Type_Float3:
        return ::rotor<float3>(displacement);
    case Type_Double3:
        return ::rotor<double3>(displacement);
    default:
        break;
    };
    FATAL() << "Unsupported voxel type "
            << "'" << stk::as_string(displacement.voxel_type()) << "'";
}

Volume circulation_density(const Volume& displacement)
{
    return map<float3, float>(rotor(displacement), norm);
}

} // namespace stk
