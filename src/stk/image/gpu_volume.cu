#include "gpu_volume.h"

#include "stk/cuda/cuda.h"
#include "stk/cuda/volume.h"

#include <algorithm>
#include <cfloat>


/*
    Finds the min/max values for a volume using multi-pass reduction.

    First pass (`volume_min_max_kernel`) takes a 3D volume as input and does a
    block-wise reduction. This step results in one min/max value per kernel block.
    If the number of blocks are greater than 1, subsequent reductions will be 
    performed in 1D (`min_max_kernel`), until only one value remains.

    The algorithm uses loop unrolling and sequential addressing within the kernel
    for a significant performance gain compared to a naive approach [1].

    [1] M. Harris, Optimizing Parallel Reduction in CUDA
*/

namespace cuda = stk::cuda;

__global__ void volume_min_max_kernel(
    const cuda::VolumePtr<float> in,
    dim3 dims,
    float2* out)
{
    #define REDUCE_2(a,b) {min(a.x, b.x), max(a.y, b.y)}

    extern __shared__ float2 shared[];

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    shared[tid].x = FLT_MAX;
    shared[tid].y = -FLT_MAX;

    if (x < dims.x &&
        y < dims.y &&
        z < dims.z) {
        shared[tid].x = in(x,y,z);
        shared[tid].y = in(x,y,z);
    }
    __syncthreads();

    // Block size is always 512 for first pass of the reduction 
    //  (with volume as input)

    // Reduces set from 2*n to n
    #define REDUCTION_STEP(n_) \
        if (tid < n_) { \
            shared[tid] = REDUCE_2(shared[tid], shared[tid+n_]);\
        } \
        __syncthreads();

    REDUCTION_STEP(256);
    REDUCTION_STEP(128);
    REDUCTION_STEP(64);
    REDUCTION_STEP(32);
    REDUCTION_STEP(16);
    REDUCTION_STEP(8);
    REDUCTION_STEP(4);
    REDUCTION_STEP(2);

    if (tid == 0) {
        // Write min/max for block to output
        out[bid] = REDUCE_2(shared[tid], shared[tid+1]);
    }

    #undef REDUCTION_STEP
    #undef REDUCE_2
}

template<int BLOCK_SIZE>
__global__ void min_max_kernel(
    unsigned int n,
    float2* in,
    float2* out)
{
    #define REDUCE_2(a,b) {min(a.x, b.x), max(a.y, b.y)}

    extern __shared__ float2 shared[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x; // global

    shared[tid] = in[i];
    __syncthreads();

    // Reduces set from 2*n to n
    #define REDUCTION_STEP(n_) \
        if (BLOCK_SIZE >= 2*n_ && tid < n_) { \
            shared[tid] = REDUCE_2(shared[tid], shared[tid+n_]); \
        } \
        __syncthreads();

    REDUCTION_STEP(512);
    REDUCTION_STEP(256);
    REDUCTION_STEP(128);
    REDUCTION_STEP(64);
    REDUCTION_STEP(32);
    REDUCTION_STEP(16);
    REDUCTION_STEP(8);
    REDUCTION_STEP(4);
    REDUCTION_STEP(2);
    
    if (tid == 0) {
        out[blockIdx.x] = REDUCE_2(shared[tid], shared[tid+1]);
    }
    
    #undef REDUCTION_STEP
    #undef REDUCE_2
}

static void run_min_max_kernel(uint32_t n_blocks, uint32_t n_threads, 
    uint32_t n, float2* d_in, float2* d_out)
{
    uint32_t shared_size = 2*sizeof(float)*n_threads;

    switch (n_threads) {
        case 1024:
            min_max_kernel<1024><<<n_blocks, n_threads, shared_size>>>(n, d_in, d_out);
            break;
        case 512:
            min_max_kernel<512><<<n_blocks, n_threads, shared_size>>>(n, d_in, d_out);
            break;
        case 256:
            min_max_kernel<256><<<n_blocks, n_threads, shared_size>>>(n, d_in, d_out);
            break;
        case 128:
            min_max_kernel<128><<<n_blocks, n_threads, shared_size>>>(n, d_in, d_out);
            break;
        case 64:
            min_max_kernel<64><<<n_blocks, n_threads, shared_size>>>(n, d_in, d_out);
            break;
        case 32:
            min_max_kernel<32><<<n_blocks, n_threads, shared_size>>>(n, d_in, d_out);
            break;
        case 16:
            min_max_kernel<16><<<n_blocks, n_threads, shared_size>>>(n, d_in, d_out);
            break;
        case  8:
            min_max_kernel<8><<<n_blocks, n_threads, shared_size>>>(n, d_in, d_out);
            break;
        case  4:
            min_max_kernel<4><<<n_blocks, n_threads, shared_size>>>(n, d_in, d_out);
            break;
        case  2:
            min_max_kernel<2><<<n_blocks, n_threads, shared_size>>>(n, d_in, d_out);
            break;
    };
}


namespace stk {
    void find_min_max(const GpuVolume& vol, float& min, float& max)
    {
        dim3 block_size{32,16,1};

        dim3 grid_size {
            (vol.size().x + block_size.x - 1) / block_size.x,
            (vol.size().y + block_size.y - 1) / block_size.y,
            (vol.size().z + block_size.z - 1) / block_size.z
        };

        // Number of blocks (or values in the active buffer)
        uint32_t n = grid_size.x * grid_size.y * grid_size.z;

        // Allocate our global buffer for holding the min/max values
        float2* d_in;
        float2* d_out;
        CUDA_CHECK_ERRORS(cudaMalloc(&d_in, 2*n*sizeof(float)));
        CUDA_CHECK_ERRORS(cudaMalloc(&d_out, 2*n*sizeof(float)));

        // TODO: Do min/max directly on texture?
        GpuVolume in_vol = vol.as_usage(gpu::Usage_PitchedPointer); 
        volume_min_max_kernel<<<grid_size, block_size, 
            uint32_t(2*sizeof(float)*block_size.x*block_size.y*block_size.z)>>>
        (
            in_vol, in_vol.size(), d_out
        );

        CUDA_CHECK_ERRORS(cudaPeekAtLastError());
        CUDA_CHECK_ERRORS(cudaDeviceSynchronize());

        while (n > 1) {
            // block_count should always be pow2 as it follows the gridsize from 
            //  previous step
            uint32_t n_threads = std::min<uint32_t>(n, 1024);
            uint32_t n_blocks = (n + n_threads - 1) / n_threads;

            CUDA_CHECK_ERRORS(cudaMemcpy(d_in, d_out, 2*n*sizeof(float), cudaMemcpyDeviceToDevice)); 
            run_min_max_kernel(n_blocks, n_threads, n, d_in, d_out);

            CUDA_CHECK_ERRORS(cudaPeekAtLastError());
            CUDA_CHECK_ERRORS(cudaDeviceSynchronize());

            n = n_blocks;
        }

        float2 min_max;
        CUDA_CHECK_ERRORS(cudaMemcpy(&min_max, d_out, 2*sizeof(float), cudaMemcpyDeviceToHost));

        min = min_max.x;
        max = min_max.y;

        CUDA_CHECK_ERRORS(cudaFree(d_in));
        CUDA_CHECK_ERRORS(cudaFree(d_out));
    }
}
