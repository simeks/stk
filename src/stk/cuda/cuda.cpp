#include "cuda.h"
#include "stk/common/assert.h"
#include "stk/common/error.h"
#include "stk/common/log.h"
#include "stk/image/gpu_volume.h"

#ifdef STK_USE_CUDA

#include <cstring>
#include <cuda_runtime.h>

namespace stk {
namespace cuda {
    TextureObject::TextureObject(const GpuVolume& vol, const cudaTextureDesc& tex_desc) : 
        _vol(vol)
    {
        ASSERT(vol.valid());
        ASSERT(vol.usage() == gpu::Usage_Texture);

        cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = vol.array_ptr();

        // TODO: Explore cost of creating/destroying texture/surface objects
        CUDA_CHECK_ERRORS(cudaCreateTextureObject(&_obj, &res_desc, &tex_desc, nullptr));
    }
    TextureObject::~TextureObject()
    {
        CUDA_CHECK_ERRORS(cudaDestroyTextureObject(_obj));
    }

    TextureObject::operator cudaTextureObject_t() const 
    {
        return _obj;
    }

    SurfaceObject::SurfaceObject(const GpuVolume& vol) : _vol(vol)
    {
        ASSERT(vol.valid());
        ASSERT(vol.usage() == gpu::Usage_Texture);

        cudaResourceDesc desc;
        memset(&desc, 0, sizeof(desc));
        desc.resType = cudaResourceTypeArray;
        desc.res.array.array = vol.array_ptr();

        CUDA_CHECK_ERRORS(cudaCreateSurfaceObject(&_obj, &desc));
    }
    SurfaceObject::~SurfaceObject()
    {
        CUDA_CHECK_ERRORS(cudaDestroySurfaceObject(_obj));
    }
    SurfaceObject::operator cudaSurfaceObject_t() const
    {
        return _obj;
    }


    void init()
    {
        int n = device_count();
        if (n == 0) {
            LOG(Warning) << "[CUDA] No CUDA enabled devices found";
            return;
        }

        for (int i = 0; i < n; ++i) {
            cudaDeviceProp prop;
            CUDA_CHECK_ERRORS(cudaGetDeviceProperties(&prop, i));

            LOG(Info) << "[CUDA] Device: " << i << " name: " << prop.name;
        }
        set_device(0);
    }
    int device_count()
    {
        int n;
        CUDA_CHECK_ERRORS(cudaGetDeviceCount(&n));
        return n;
    }
    void set_device(int device_id)
    {
        CUDA_CHECK_ERRORS(cudaSetDevice(device_id));
    }
    int device()
    {
        int device_id;
        CUDA_CHECK_ERRORS(cudaGetDevice(&device_id));
        return device_id;
    }
    void reset_device()
    {
        CUDA_CHECK_ERRORS(cudaDeviceReset());
    }

} // namespace cuda
} // namespace stk

#endif // STK_USE_CUDA
