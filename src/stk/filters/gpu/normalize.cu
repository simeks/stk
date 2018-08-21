#include "normalize.h"

#include "stk/common/error.h"
#include "stk/cuda/cuda.h"
#include "stk/cuda/volume.h"
#include "stk/image/gpu_volume.h"

namespace cuda = stk::cuda;


__global__ void normalize_texture(
    cudaTextureObject_t src,
    dim3 dims,
    float src_min,
    float src_range,
    float min,
    float range,
    cudaSurfaceObject_t out)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= dims.x ||
        y >= dims.y ||
        z >= dims.z)
    {
        return;
    }

    float s = tex3D<float>(src, x+0.5f, y+0.5f, z+0.5f);
    float v = range * (s - src_min) / src_range + min;

    surf3Dwrite(
        v,
        out,
        x*sizeof(float), y, z
    );
}

__global__ void normalize_pitched(
    cuda::VolumePtr<float> src,
    dim3 dims,
    float src_min,
    float src_range,
    float min,
    float range,
    cuda::VolumePtr<float> out)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= dims.x ||
        y >= dims.y ||
        z >= dims.z)
    {
        return;
    }

    out(x,y,z) = range * (src(x,y,z) - src_min) / src_range + min;
}

namespace stk {
namespace gpu {
    GpuVolume normalize(
        const GpuVolume& src, 
        float min, 
        float max, 
        GpuVolume* out,
        const dim3& block_size
    )
    {
        FATAL_IF(src.voxel_type() != stk::Type_Float)
            << "Unsupported voxel format";
        
        float src_min, src_max;
        stk::find_min_max(src, src_min, src_max);

        dim3 dims = src.size();

        GpuVolume dest;
        if (!out) {
            dest.allocate(src.size(), src.voxel_type(), src.usage());
            out = &dest;
        }
        ASSERT(out->size() == dims);

        out->copy_meta_from(src);

        float range = float(max - min);
        float src_range = float(src_max - src_min);

        dim3 grid_size {
            (dims.x + block_size.x - 1) / block_size.x,
            (dims.y + block_size.y - 1) / block_size.y,
            (dims.z + block_size.z - 1) / block_size.z
        };
                
        if (src.usage() == Usage_Texture) {
            cudaTextureDesc tex_desc;
            memset(&tex_desc, 0, sizeof(tex_desc));
            tex_desc.addressMode[0] = cudaAddressModeClamp;
            tex_desc.addressMode[1] = cudaAddressModeClamp;
            tex_desc.addressMode[2] = cudaAddressModeClamp;
            tex_desc.addressMode[2] = cudaAddressModeClamp;
            tex_desc.filterMode = cudaFilterModePoint;
            
            normalize_texture<<<grid_size, block_size>>>(
                cuda::TextureObject(src, tex_desc),
                dims,
                src_min,
                src_range,
                min,
                range,
                cuda::SurfaceObject(*out)
            );
        }
        else {
            normalize_pitched<<<grid_size, block_size>>>(
                src,
                dims,
                src_min,
                src_range,
                min,
                range,
                *out
            );
        }
        return *out;
    }
}
}
