#include "catch.hpp"

#include <stk/cuda/cuda.h>
#include <stk/cuda/stream.h>
#include <stk/image/volume.h>

using namespace stk;

TEST_CASE("cuda", "[cuda]")
{
    REQUIRE_NOTHROW(cuda::init());

    int device_count = 0;
    REQUIRE_NOTHROW(device_count = cuda::device_count());
    REQUIRE(device_count > 0);

    for (int i = 0; i < device_count; ++i) {
        REQUIRE_NOTHROW(cuda::set_device(i));
        REQUIRE(cuda::device() == i);
    }
}
TEST_CASE("cuda_pinned_memory", "[cuda]")
{
    REQUIRE_NOTHROW(cuda::init());

    dim3 dims[] = {
        {32,32,32},
        {64,64,64},
        {128,128,128},
        {256,256,256},
        {512,512,512}
    };

    for (int i = 0; i < 5; ++i) {
        Volume vol;
        REQUIRE_NOTHROW(vol.allocate(dims[i], Type_Float, Usage_Pinned));
        REQUIRE(vol.ptr());
        REQUIRE_NOTHROW(vol.release());

        REQUIRE_NOTHROW(vol.allocate(dims[i], Type_Float, Usage_Mapped));
        REQUIRE(vol.ptr());
        REQUIRE_NOTHROW(vol.release());

        REQUIRE_NOTHROW(vol.allocate(dims[i], Type_Float, Usage_WriteCombined));
        REQUIRE(vol.ptr());
        REQUIRE_NOTHROW(vol.release());
    }
}
TEST_CASE("cuda_stream", "[cuda]")
{
    cuda::Stream stream;
    REQUIRE(((cudaStream_t)stream) != 0);

    bool callback_triggered = false;
    REQUIRE_NOTHROW(stream.add_callback(
        [&](cudaStream_t, int){ 
            callback_triggered = true; 
        }));
    REQUIRE_NOTHROW(stream.synchronize());
    REQUIRE(stream.query() == true);
    REQUIRE(callback_triggered == true);
}
TEST_CASE("cuda_event", "[cuda]")
{
    cuda::Stream stream;
    REQUIRE(((cudaStream_t)stream) != 0);

    float data[256];
    float* d_data;
    REQUIRE(cudaMalloc(&d_data, 256*sizeof(float)) == cudaSuccess);

    cuda::Event evt0;
    cuda::Event evt1;

    evt0.record(stream);
    REQUIRE(cudaMemcpyAsync(d_data, data, 256*sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess);
    evt1.record(stream);
    evt1.synchronize();
    
    float ms = cuda::Event::elapsed(evt0, evt1);
    REQUIRE(ms > 0.0f);
}


