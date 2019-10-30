#include "cuda.h"
#include "volume.h"

#include "stk/image/gpu_volume.h"

#include <cstring>

#ifdef STK_USE_CUDA

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


} // namespace cuda
} // namespace stk

#endif // STK_USE_CUDA
