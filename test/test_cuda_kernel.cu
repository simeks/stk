#include "catch.hpp"

#include <cuda_runtime.h>

#include <stk/cuda/cuda.h>
#include <stk/cuda/volume.h>
#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>

#include "test_util.h"

using namespace stk;

namespace {
    const uint32_t W = 20;
    const uint32_t H = 30;
    const uint32_t D = 40;
}

template<typename T>
__global__ void copy_kernel(cuda::VolumePtr<T> in, cuda::VolumePtr<T> out)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= W || y >= H || z >= D) {
        return;
    }

    out(x,y,z) = in(x,y,z);
}

template<typename T>
__global__ void copy_texture_kernel(cudaTextureObject_t in, cudaSurfaceObject_t out)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= W || y >= H || z >= D) {
        return;
    }

    T v = tex3D<T>(in, x + 0.5f, y + 0.5f, z + 0.5f);
    surf3Dwrite(v, out, x*sizeof(T), y, z);
}

TEST_CASE("cuda_copy_kernel", "[cuda]")
{
    #define TEST_TYPE(T) \
        SECTION(#T) { \
            T* test_data = new T[W*H*D]; \
            TestDataGenerator<T>::run(test_data, W, H, D); \
            VolumeHelper<T> in({W,H,D}, test_data); \
            GpuVolume gpu_in(in, gpu::Usage_PitchedPointer); \
            GpuVolume gpu_out(gpu_in.size(), gpu_in.voxel_type(), gpu::Usage_PitchedPointer); \
            dim3 block_size{8,8,1}; \
            dim3 grid_size { \
                (W + block_size.x - 1) / block_size.x, \
                (H + block_size.y - 1) / block_size.y, \
                (D + block_size.z - 1) / block_size.z \
            }; \
            copy_kernel<T><<<grid_size, block_size>>>( \
                gpu_in, \
                gpu_out \
            ); \
            CUDA_CHECK_ERRORS(cudaDeviceSynchronize()); \
            Volume out = gpu_out.download(); \
            REQUIRE(compare_volumes<T>(in, out)); \
            delete [] test_data;\
        }

    TEST_TYPE(char);
    TEST_TYPE(char2);
    TEST_TYPE(char4);

    TEST_TYPE(uint8_t);
    TEST_TYPE(uchar2);
    TEST_TYPE(uchar4);
    
    TEST_TYPE(short);
    TEST_TYPE(short2);
    TEST_TYPE(short4);

    TEST_TYPE(uint16_t);
    TEST_TYPE(short2);
    TEST_TYPE(short4);

    TEST_TYPE(int);
    TEST_TYPE(int2);
    TEST_TYPE(int4);

    TEST_TYPE(uint32_t);
    TEST_TYPE(uint2);
    TEST_TYPE(uint4);

    TEST_TYPE(float);
    TEST_TYPE(float2);
    TEST_TYPE(float4);

    #undef TEST_TYPE
}

TEST_CASE("cuda_copy_texture_kernel", "[cuda]")
{
    #define TEST_TYPE(T) \
        SECTION(#T) { \
            T* test_data = new T[W*H*D]; \
            TestDataGenerator<T>::run(test_data, W, H, D); \
            VolumeHelper<T> in({W,H,D}, test_data); \
            GpuVolume gpu_in(in, gpu::Usage_Texture); \
            GpuVolume gpu_out(gpu_in.size(), gpu_in.voxel_type(), gpu::Usage_Texture); \
            cudaTextureDesc tex_desc; \
            memset(&tex_desc, 0, sizeof(tex_desc)); \
            tex_desc.addressMode[0] = cudaAddressModeClamp; \
            tex_desc.addressMode[1] = cudaAddressModeClamp; \
            tex_desc.addressMode[2] = cudaAddressModeClamp; \
            tex_desc.filterMode = cudaFilterModePoint; \
            cuda::TextureObject in_obj(gpu_in, tex_desc); \
            cuda::SurfaceObject out_obj(gpu_out); \
            dim3 block_size{8,8,1}; \
            dim3 grid_size { \
                (W + block_size.x - 1) / block_size.x, \
                (H + block_size.y - 1) / block_size.y, \
                (D + block_size.z - 1) / block_size.z \
            }; \
            copy_texture_kernel<T><<<grid_size, block_size>>>( \
                in_obj, \
                out_obj \
            ); \
            CUDA_CHECK_ERRORS(cudaDeviceSynchronize()); \
            VolumeHelper<T> out = gpu_out.download(); \
            REQUIRE(compare_volumes<T>(in, out)); \
            delete [] test_data; \
        }
    
    TEST_TYPE(char);
    TEST_TYPE(char2);
    TEST_TYPE(char4);

    TEST_TYPE(uint8_t);
    TEST_TYPE(uchar2);
    TEST_TYPE(uchar4);
    
    TEST_TYPE(short);
    TEST_TYPE(short2);
    TEST_TYPE(short4);

    TEST_TYPE(uint16_t);
    TEST_TYPE(short2);
    TEST_TYPE(short4);

    TEST_TYPE(int);
    TEST_TYPE(int2);
    TEST_TYPE(int4);

    TEST_TYPE(uint32_t);
    TEST_TYPE(uint2);
    TEST_TYPE(uint4);

    TEST_TYPE(float);
    TEST_TYPE(float2);
    TEST_TYPE(float4);

    #undef TEST_TYPE
}

