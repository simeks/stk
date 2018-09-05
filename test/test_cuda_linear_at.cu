#include "catch.hpp"

#include <cuda_runtime.h>

#include <stk/cuda/cuda.h>
#include <stk/cuda/volume.h>
#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>

#include "test_util.h"

using namespace stk;

template<typename T>
__global__ void linear_at_border_kernel(cuda::VolumePtr<T> in, cuda::VolumePtr<T> out, dim3 dims, float3 offset)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= dims.x || y >= dims.y || z >= dims.z) {
        return;
    }

    out(x,y,z) = cuda::linear_at_border(in, dims, x+offset.x, y+offset.y, z+offset.z);
}

TEST_CASE("cuda_linear_at", "[cuda]")
{
    dim3 dims{2,2,2};
    
    VolumeFloat in(dims, 1.0f);
    in(0,0,0) = 0.0f;

    VolumeFloat out(dims, 0.0f);

    GpuVolume gpu_in(in);
    GpuVolume gpu_out(out);
    linear_at_border_kernel<float><<<dim3{1,1,1}, dims>>>(gpu_in, gpu_out, dims, float3{0.25f, 0.0f, 0.0f});

    gpu_out.download(out);
    REQUIRE(out(0,0,0) == Approx(0.25f));

    linear_at_border_kernel<float><<<dim3{1,1,1}, dims>>>(gpu_in, gpu_out, dims, float3{0.0f, 0.1f, 0.0f});

    gpu_out.download(out);
    REQUIRE(out(0,0,0) == Approx(0.1f));

    linear_at_border_kernel<float><<<dim3{1,1,1}, dims>>>(gpu_in, gpu_out, dims, float3{0.0f, 0.0f, 0.05f});

    gpu_out.download(out);
    REQUIRE(out(0,0,0) == Approx(0.05f));
    
    linear_at_border_kernel<float><<<dim3{1,1,1}, dims>>>(gpu_in, gpu_out, dims, float3{10.05f, 0.0f, 0.0f});

    gpu_out.download(out);
    REQUIRE(out(0,0,0) == Approx(0.0f));
    
    linear_at_border_kernel<float><<<dim3{1,1,1}, dims>>>(gpu_in, gpu_out, dims, float3{0.0f, 10.05f, 0.0f});

    gpu_out.download(out);
    REQUIRE(out(0,0,0) == Approx(0.0f));

    linear_at_border_kernel<float><<<dim3{1,1,1}, dims>>>(gpu_in, gpu_out, dims, float3{0.0f, 0.0f, 10.05f});

    gpu_out.download(out);
    REQUIRE(out(0,0,0) == Approx(0.0f));
}

