#include <cuda_runtime.h>

#include <stk/cuda/ptr.h>
#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>

#include "test_util.h"

using namespace stk;

namespace {
    const uint32_t W = 20;
    const uint32_t H = 30;
    const uint32_t D = 40;
}

__global__ void copy_kernel(cuda::VolumePtr in, cuda::VolumePtr out)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= new_dims.width ||
        y >= new_dims.height ||
        z >= new_dims.depth)
    {
        return;
    }

    out(x,y,z) = in(x,y,z);
}


TEST_CASE("cuda_copy_kernel", "[cuda]")
{
    SECTION("float")
    {
        float test_data[W*H*D];
        TestDataGenerator::run(test_data, W, H, D);
        
        VolumeHelper<float> in({W,H,D}, test_data);
        
        GpuVolume gpu_in(in, gpu::Usage_PitchedPointer);
        GpuVolume gpu_out(gpu_in.size(), gpu_in.voxel_type(), gpu::Usage_PitchedPointer);

        dim3 block_size{8,8,1};
        dim3 grid_size
        {
            (W + block_size.x - 1) / block_size.x,
            (H + block_size.y - 1) / block_size.y,
            (D + block_size.z - 1) / block_size.z
        };
        
        copy_kernel<<<grid_size, block_size>>>(
            gpu_in.pitched_ptr(),
            gpu_out.pitched_ptr()
        );

        Volume out = gpu_out.download();

        REQUIRE(compare_volumes(in, out));
    }
}
