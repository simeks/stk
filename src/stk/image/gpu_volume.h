#pragma once

#ifdef STK_USE_CUDA

#include "dim3.h"
#include "types.h"

#include <channel_descriptor.h>
#include <memory>
#include <stdint.h>

namespace stk
{
    class Volume;
    struct Range;

    namespace cuda
    {
        class Stream;
    }

    namespace gpu
    {
        enum Usage
        {
            Usage_PitchedPointer,
            Usage_Texture
        };
    }

    struct GpuVolumeData
    {
        GpuVolumeData();
        ~GpuVolumeData();

        gpu::Usage usage;
        cudaChannelFormatDesc format_desc;

        cudaArray* array_ptr;
        cudaPitchedPtr pitched_ptr;
    };

    class GpuVolume
    {
    public:
        GpuVolume();

        // @remark This does not copy the data, use clone if you want a separate copy.
        GpuVolume(const GpuVolume& other);
        // @remark This does not copy the data, use clone if you want a separate copy.
        GpuVolume& operator=(const GpuVolume& other);

        // The usage parameter specified whether this volume should be accessed
        //  as a pitched pointer, i.e.
        //      void kernel(ptr) { x = ptr[0]; }
        //  or a texture, where you'll have to bind it before using it, i.e.
        //      void kernel(ptr) { x = tex3D(vol, px, py, pz); }
        GpuVolume(const dim3& size, Type voxel_type,
            gpu::Usage usage = gpu::Usage_PitchedPointer);

        // Creates a new reference to a region within an existing volume
        // There's a chance that the resulting volume does not contain contiguous memory when
        //  created using this constructor. Use ptr() with caution and see `is_contiguous`.
        // @remark This does not copy the data, use clone if you want a separate copy.
        GpuVolume(const GpuVolume& other, const Range& x, const Range& y, const Range& z);
        ~GpuVolume();

        // Allocates volume memory for a volume with specified parameters
        // Will release any previously allocated memory.
        void allocate(const dim3& size, Type voxel_type,
            gpu::Usage usage = gpu::Usage_PitchedPointer);

        // Releases the volume data
        void release();

        // Clones this volume, creating a new volume with same content, type, and usage
        GpuVolume clone() const;

        // Clones this volume, creating a new volume with same content, type, and usage
        //  (non-blocking). The returned object will hold the allocated memory, but the
        //  data within it is not ready until the stream has completed the copying.
        GpuVolume clone(const cuda::Stream& stream) const;

        // Clones this volume, creating a new volume with same content and type
        // usage : Specifies usage for the new volume
        GpuVolume clone_as(gpu::Usage usage) const;

        // Clones this volume, creating a new volume with same content and type
        //  (non-blocking). The returned object will hold the allocated memory, but the
        //  data within it is not ready until the stream has completed the copying.
        // usage : Specifies usage for the new volume
        GpuVolume clone_as(gpu::Usage usage, const cuda::Stream& stream) const;

        // Copies the data from the given volume
        // This assumes that both volumes are of the same size and data type
        // Compared to using clone, this does not perform any memory allocations.
        void copy_from(const GpuVolume& other);

        // Copies the data from the given volume (non-blocking)
        // This assumes that both volumes are of the same size and data type
        // Compared to using clone, this does not perform any memory allocations.
        void copy_from(const GpuVolume& other, const cuda::Stream& stream);

        // Returns true if the volume is valid (allocated and ready to use), false if not
        bool valid() const;

        // Returns the size of the volume
        dim3 size() const;

        void set_origin(const float3& origin);
        void set_spacing(const float3& spacing);
        void set_direction(const Matrix3x3f& direction);
        void set_direction(const std::initializer_list<float> direction);

        const float3& origin() const;
        const float3& spacing() const;
        const Matrix3x3f& direction() const;
        const Matrix3x3f& inverse_direction() const;

        // Copies meta data (origin, spacing, ...) from the provided volume.
        void copy_meta_from(const GpuVolume& other);

        // Tries to extract the voxel-type for this volume
        // See stk::Type
        Type voxel_type() const;

        // Creates a new volume on the GPU side and uploads the given volume into it.
        // For usage parameter, see constructor GpuVolume(dim3, Type, Usage)
        GpuVolume(const Volume& vol, gpu::Usage usage = gpu::Usage_PitchedPointer);

        // Creates a new volume on the GPU side and uploads the given volume into it.
        // For usage parameter, see constructor GpuVolume(dim3, Type, Usage)
        // (non-blocking). The constructor will allocate the backing memory and
        //  invoke the data transfer but the data will not be ready for use until
        //  the provided stream has completed.
        GpuVolume(const Volume& vol,
                  const cuda::Stream& stream,
                  gpu::Usage usage = gpu::Usage_PitchedPointer);

        // Downloads this volume to a new volume
        // @return Handle to newly created volume
        Volume download() const;

        // Downloads this volume to a new volume (non-blocking).
        // Returned volume will contain allocated memory, but the data is not ready
        //  until the transfer within the stream has completed.
        // GpuVolume::download(Volume&) is prefered for async transfers, it also
        //  provides the possibility for page-locked memory within volumes.
        // @return Handle to newly created volume
        Volume download(const cuda::Stream& stream) const;

        // Downloads this volume to given volume
        // @remark Requires both volumes to be of same size and type
        void download(Volume& vol) const;

        // Downloads this volume to given volume (non-blocking)
        // @remark Requires both volumes to be of same size and type
        void download(Volume& vol, const cuda::Stream& stream) const;

        // Uploads the given volume into this gpu volume
        // @remark Requires both volumes to be of same size and type
        void upload(const Volume& vol);

        // Uploads the given volume into this gpu volume (non-blocking)
        // @remark Requires both volumes to be of same size and type
        void upload(const Volume& vol, const cuda::Stream& stream);

        // Returns the usage flags for this volume
        // Requires volume to be allocated
        gpu::Usage usage() const;

        // Returns a volume with the specific usage flag.
        // This will either (1) return a reference to this volume, or (2) create
        //     a new copy on the GPU and return a reference to the copy.
        // (1) happends when requested usage matches current volume. Otherwise
        //     we have no other choice than to create a new volume and copy (2).
        GpuVolume as_usage(gpu::Usage usage) const;

        // Direct access to underlying data (device ptr)
        // Only valid when usage is set to texture or surface
        cudaArray* array_ptr() const;

        // Direct access to underlying data (device ptr)
        // Only valid when usage is set to pitched pointer
        cudaPitchedPtr pitched_ptr() const;

        // Creates a new reference to a region within an existing volume
        // There's a chance that the resulting volume does not contain contiguous memory when
        //  created using this constructor.
        // @remark This does not copy the data, use clone if you want a separate copy.
        GpuVolume operator()(const Range& x, const Range& y, const Range& z);

    private:
        std::shared_ptr<GpuVolumeData> _data;

        dim3 _size;
        float3 _origin;
        float3 _spacing;
        Matrix3x3f _direction;
        Matrix3x3f _inverse_direction;

        cudaPitchedPtr _ptr; // For pitched pointer usage only
    };

    namespace gpu
    {
        // Tries to extract the voxel-type for this volume
        Type voxel_type(const GpuVolume& vol);
    }

    // Finds the minimum and maximum values in a scalar volume.
    // Only works for float32 volumes.
    void find_min_max(const GpuVolume& vol, float& min, float& max);

}

#endif // STK_USE_CUDA
