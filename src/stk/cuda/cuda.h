#pragma once

#ifdef __CUDACC__
    #define CUDA_HOST_DEVICE __host__ __device__
#else
    #define CUDA_HOST_DEVICE
#endif

#ifdef STK_USE_CUDA

#include <cuda_runtime.h>

#include <stk/common/error.h>
#include <stk/image/gpu_volume.h>

#define CUDA_CHECK_ERRORS(val) \
    if (val != cudaSuccess) { \
        FATAL() << "[CUDA] " << cudaGetErrorString(val) << "(code=" << val << ")"; \
    }

namespace stk
{
    class GpuVolume;

    namespace cuda
    {
        // Wrapper around cudaTextureObject_t, will automatically destroy
        //  the object when going out of scope.
        class TextureObject
        {
        public:
            TextureObject(const GpuVolume& vol, const cudaTextureDesc& tex_desc);
            ~TextureObject();

            operator cudaTextureObject_t() const;

        private:
            TextureObject(const TextureObject&);
            TextureObject& operator=(const TextureObject&);

            cudaTextureObject_t _obj;

            // Keep a reference to the volume to make sure it won't get destroyed
            GpuVolume _vol;
        };

        // Wrapper around cudaSurfaceObject_t, will automatically destroy
        //  the object when going out of scope.
        class SurfaceObject
        {
        public:
            SurfaceObject(const GpuVolume& vol);
            ~SurfaceObject();

            operator cudaSurfaceObject_t() const;

        private:
            SurfaceObject(const SurfaceObject&);
            SurfaceObject& operator=(const SurfaceObject&);

            cudaSurfaceObject_t _obj;

            // Keep a reference to the volume to make sure it won't get destroyed
            GpuVolume _vol;
        };

        // Initializes CUDA
        void init();

        // Returns the number of CUDA-enabled devices
        int device_count();

        // Sets the active cuda device
        void set_device(int device_id);

        // Returns the index to the currently active device
        int device();

        // Resets the current device
        void reset_device();
    }
}
#endif // STK_USE_CUDA
