#pragma once

#include <stk/common/assert.h>

#include "dim3.h"
#include "types.h"


#include <memory>

namespace stk
{
    enum BorderMode
    {
        Border_Constant, // Zero padding outside volume
        Border_Replicate
    };

#ifdef STK_USE_CUDA
    struct GpuVolume;
#endif // STK_USE_CUDA

    struct VolumeData
    {
        VolumeData();
        VolumeData(size_t size);
        ~VolumeData();

        uint8_t* data;
        size_t size;
    };

    // Volume is essentially a wrapper around a reference counted VolumeData. It represents
    //  a three-dimensional matrix with a specified element type.
    //
    // Usage:
    //  Create and allocate memory for a new float volume of size WxHxD: 
    //      Volume vol({W, H, D}, Type_Float);
    //  Or using an already existing object:
    //      vol.allocate({W, H, D}, Type_Float);
    //  this releases any previously allocated data.
    //  The data is automatically released whenever the data reference count reaches 0
    //  (e.g. the vol object goes out of scope and no other references exists). The release
    //  method can be used to explicitly release the data.
    //  
    // Copying the object, either through the assignment operator or copy constructor
    //  will create a new reference to the already existing data and only the header
    //  info is copied. Therefore, multiple volume objects may reference to the same data.
    //  To create a deep copy of the data, use the clone() method, e.g.
    //      Volume real_copy = vol.clone()
    //
    // TODO: Custom memory allocator

    class Volume
    {
    public:
        Volume();
        // If a data pointer is specified, the volume copies that data into its newly
        //  allocated memory.
        Volume(const dim3& size, Type voxel_type, void* data = NULL);
        ~Volume();

        // Note: Resets spacing and origin
        void allocate(const dim3& size, Type voxel_type);

        // Release any allocated data the volume is holding
        // This makes the volume object invalid
        // Note: 
        //     Resets spacing and origin
        void release();

        // Clones this volume
        Volume clone() const;

        // Copies the data from the given volume
        // This assumes that both volumes are of the same size and data type
        // Compoared to using clone, this does not perform any memory allocations.
        // Also copies the meta data (origin and spacing) of other volume.
        void copy_from(const Volume& other);

        // Attempts to convert this volume to the specified format,
        //     If this volume already is of the specified format it will just return itself.
        //     If not a new converted version will be allocated and returned.
        // Note:
        //     You can only convert between volumes with the same number of components per voxel.
        //     I.e. you cannot convert from Float3 to Int2.
        Volume as_type(Type type) const;

        // Returns true if the volume is allocated and ready for use
        bool valid() const;

        // Raw pointer to the volume data
        void* ptr();

        // Raw pointer to the volume data
        void const* ptr() const;

        Type voxel_type() const;
        const dim3& size() const;

        void set_origin(const float3& origin);
        void set_spacing(const float3& spacing);

        const float3& origin() const;
        const float3& spacing() const;

        // @remark This does not copy the data, use clone if you want a separate copy.
        Volume(const Volume& other);
        Volume& operator=(const Volume& other);

    #ifdef STK_USE_CUDA
        // Creates a new volume on the CPU side and downloads the given volume from the gpu into it.
        Volume(const GpuVolume& gpu_volume);

        // Uploads this volume to a newly allocated GPU volume
        // @remark Requires both volumes to be of same size and type
        // @return Handle to newly created GPU volume
        GpuVolume upload() const;

        // Uploads this volume to given GPU volume
        // @remark Requires both volumes to be of same size and type
        void upload(const GpuVolume& gpu_volume) const;

        // Downloads the given volume into this volume
        // @remark Requires both volumes to be of same size and type
        void download(const GpuVolume& gpu_volume);
    #endif // STK_USE_CUDA


    protected:
        std::shared_ptr<VolumeData> _data;
        void* _ptr; // Pointer to a location in _data
        size_t _stride; // Size of a single row in bytes

        dim3 _size;
        Type _voxel_type;

        float3 _origin; // Origin in world coordinates
        float3 _spacing; // Size of a voxel
    };

    template<typename T>
    class VolumeHelper : public Volume
    {
    public:
        typedef T TVoxelType;
        
        // Creates a null (invalid) volume
        VolumeHelper();
        // Converts the given volume if the voxel type does not match
        VolumeHelper(const Volume& other);
        // Creates a new volume of the specified size
        VolumeHelper(const dim3& size);
        // Creates a new volume of the specified size and initializes it with the given value
        explicit VolumeHelper(const dim3& size, const T& value);
        // Creates a new volume and copies the given data
        explicit VolumeHelper(const dim3& size, T* value);
        ~VolumeHelper();

        // Fills the volume with the specified value
        void fill(const T& value);

        // Returns value at 
        T at(int x, int y, int z, BorderMode border_mode) const;
        T at(int3 p, BorderMode border_mode) const;

        T linear_at(float x, float y, float z, BorderMode border_mode) const;
        T linear_at(float3 p, BorderMode border_mode) const;

        VolumeHelper& operator=(const VolumeHelper& other);
        VolumeHelper& operator=(const Volume& other);

        const T& operator()(int x, int y, int z) const;
        T& operator()(int x, int y, int z);
        
        const T& operator()(const int3& p) const;
        T& operator()(const int3& p);

        // Offset in bytes to the specified element
        size_t offset(int x, int y, int z) const;

    };

    typedef VolumeHelper<uint8_t> VolumeUChar;
    typedef VolumeHelper<uchar2>  VolumeUChar2;
    typedef VolumeHelper<uchar3>  VolumeUChar3;
    typedef VolumeHelper<uchar4>  VolumeUChar4;

    typedef VolumeHelper<float>  VolumeFloat;
    typedef VolumeHelper<float2> VolumeFloat2;
    typedef VolumeHelper<float3> VolumeFloat3;
    typedef VolumeHelper<float4> VolumeFloat4;

    typedef VolumeHelper<double>  VolumeDouble;
    typedef VolumeHelper<double2> VolumeDouble2;
    typedef VolumeHelper<double3> VolumeDouble3;
    typedef VolumeHelper<double4> VolumeDouble4;
}

#include "volume.inl"