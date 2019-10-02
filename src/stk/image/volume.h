#pragma once

#include <stk/common/assert.h>
#include <stk/math/matrix3x3f.h>

#include "dim3.h"
#include "types.h"

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

namespace stk
{
    enum BorderMode
    {
        Border_Constant,  // Zero padding outside volume
        Border_Replicate, // Clamp to edge
        Border_Mirror,    // Mirror
        Border_Cyclic,    // Wrap indices around
    };

    // Flags for allocation of the backing memory for the volumes. Currently only
    //  with CUDA. These flags allows for more efficient transfer of data between
    //  CPU and GPU. However, use sparingly, as excessive use may degrade system
    //  performance.
    enum Usage
    {
    #ifdef STK_USE_CUDA
        Usage_Pinned = 1,       // Pinned, or page-locked memory, for async operations with CUDA.
        Usage_Mapped = 2,       // Mapped to CUDA address space
        Usage_WriteCombined = 4 // Slow on read on CPUs but possibly a quicker transfer across PCIe
    #endif // STK_USE_CUDA
    };

    // [begin, end)
    struct Range
    {
        int begin, end;
    };

    struct VolumeData
    {
        VolumeData();
        VolumeData(size_t size, uint32_t flags);
        ~VolumeData();

        uint8_t* data;
        size_t size;

        uint32_t flags;
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
        // flags : See Usage
        Volume(const dim3& size, Type voxel_type, const void* data = nullptr, uint32_t flags = 0);

        // Creates a new reference to a region within an existing volume
        // There's a chance that the resulting volume does not contain contiguous memory when
        //  created using this constructor. Use ptr() with caution and see `is_contiguous`.
        // @remark This does not copy the data, use clone if you want a separate copy.
        Volume(const Volume& other, const Range& x, const Range& y, const Range& z);

        ~Volume();

        // Note: Resets spacing and origin
        void allocate(const dim3& size, Type voxel_type, uint32_t flags = 0);

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
        // Access through raw pointer should be avoided as the data can be non-contiguous.
        //  Use `is_contiguous` to verify.
        void* ptr();

        // Raw pointer to the volume data
        // Access through raw pointer should be avoided as the data can be non-contiguous.
        //  Use `is_contiguous` to verify.
        void const* ptr() const;

        Type voxel_type() const;
        const dim3& size() const;

        // Set and get information about the image space.
        //  * `origin` is a point in image space, denoting the physical
        //    location of the voxel with indices (0, 0, 0).
        //  * `spacing` is the physical distance between samples in the
        //    orthogonal voxel grid.
        //  * `direction` is a non-singular matrix representing an
        //    affine transform that maps from the orthogonal system of the
        //    voxel grid to the physical space.
        //
        //  NOTE: the direction matrix is not necessarily orghogonal and
        //        its columns may not be unit vectors, implying that the
        //        actual physical distance between samples can have two
        //        components, one due to `spacing` and another due to
        //        the shear component of the direction matrix.
        void set_origin(const float3& origin);
        void set_spacing(const float3& spacing);
        void set_direction(const Matrix3x3f& direction);
        void set_direction(const std::initializer_list<float> direction);

        const float3& origin() const;
        const float3& spacing() const;
        const Matrix3x3f& direction() const;

        // Convert between voxel indices and spatial coordinates
        const float3 index2point(const float3& index) const;
        const float3 index2point(const int3& index) const;
        const float3 point2index(const float3& point) const;
        const float3 point2index(const int3& point) const;

        // Strides for x, y, z
        const size_t* strides() const;

        // Copies meta data (origin, spacing, ...) from the provided volume.
        void copy_meta_from(const Volume& other);

        // Returns true if the volume resides contigiuously in memory.
        // Volumes are typically contiguous, however, if the volume references
        //  a sub region of a larger volume, the memory is not contiguous.
        bool is_contiguous() const;

        // Creates a new reference to a region within an existing volume
        // There's a chance that the resulting volume does not contain contiguous memory when
        //  created using this constructor. Use ptr() with caution and see `is_contiguous`.
        // @remark This does not copy the data, use clone if you want a separate copy.
        Volume operator()(const Range& x, const Range& y, const Range& z);

        // @remark This does not copy the data, use clone if you want a separate copy.
        Volume(const Volume& other);
        // @remark This does not copy the data, use clone if you want a separate copy.
        Volume& operator=(const Volume& other);

        // Handle metadata (not threadsafe)
        // Metadata are key-value mappings betweeen string values,
        //  stored in a dictionary. A copy-on-write mechanism for
        //  metadata is used when copying volumes, and an own deep copy
        //  of metadata is created only when the copied volume changes
        //  the dictionary.
        std::vector<std::string> get_metadata_keys(void) const;
        std::string get_metadata(const std::string& key) const;
        void set_metadata(const std::string& key, const std::string& value);

    protected:
        using MetaDataDictionary = std::map<std::string, std::string>;

        std::shared_ptr<VolumeData> _data;
        void* _ptr; // Pointer to a location in _data

        // Strides in allocated volume memory (in bytes)
        // _step[0] : Size of element (x)
        // _step[1] : Size of one row (y)
        // _step[2] : Size of one slice (z)
        size_t _strides[3];

        dim3 _size;
        Type _voxel_type;

        float3 _origin; // Origin in world coordinates
        float3 _spacing; // Size of a voxel
        Matrix3x3f _direction; // Cosine directions of the axes
        Matrix3x3f _inverse_direction;

        bool _contiguous;

        std::shared_ptr<MetaDataDictionary> _metadata;
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
        // Copy constructor
        VolumeHelper(const VolumeHelper&) = default;

        // Creates a new volume of the specified size and initializes it with the given value
        explicit VolumeHelper(const dim3& size, const T& value);
        // Creates a new volume and copies the given data
        explicit VolumeHelper(const dim3& size, T* value);

        // Creates a new reference to a region within an existing volume
        // There's a chance that the resulting volume does not contain contiguous memory when
        //  created using this constructor. Use ptr() with caution and see `is_contiguous`.
        // @remark This does not copy the data, use clone if you want a separate copy.
        VolumeHelper(const VolumeHelper& other, const Range& x, const Range& y, const Range& z);
        ~VolumeHelper();

        // Note: Resets spacing and origin
        void allocate(const dim3& size);

        // Fills the volume with the specified value
        void fill(const T& value);

        // Returns value at
        T at(int x, int y, int z, BorderMode border_mode) const;
        T at(int3 p, BorderMode border_mode) const;

        // Indices in the voxel grid
        T linear_at(float x, float y, float z, BorderMode border_mode) const;
        T linear_at(float3 p, BorderMode border_mode) const;

        // Point coordinates in image space
        T linear_at_point(float x, float y, float z, BorderMode border_mode) const;
        T linear_at_point(float3 p, BorderMode border_mode) const;

        VolumeHelper& operator=(const VolumeHelper& other);
        VolumeHelper& operator=(const Volume& other);

        const T& operator()(int x, int y, int z) const;
        T& operator()(int x, int y, int z);

        const T& operator()(const int3& p) const;
        T& operator()(const int3& p);

        // Creates a new reference to a region within an existing volume
        // There's a chance that the resulting volume does not contain contiguous memory when
        //  created using this constructor. Use ptr() with caution and see `is_contiguous`.
        // @remark This does not copy the data, use clone if you want a separate copy.
        VolumeHelper operator()(const Range& x, const Range& y, const Range& z);

        // Offset in bytes to the specified element
        size_t offset(int x, int y, int z) const;

    };

    // Finds the minimum and maximum values in a scalar volume.
    // Does not work for multi-channel volumes.
    template<typename T>
    void find_min_max(const VolumeHelper<T>& vol, T& min, T& max);

    typedef VolumeHelper<char>      VolumeChar;
    typedef VolumeHelper<char2>     VolumeChar2;
    typedef VolumeHelper<char3>     VolumeChar3;
    typedef VolumeHelper<char4>     VolumeChar4;

    typedef VolumeHelper<uint8_t>   VolumeUChar;
    typedef VolumeHelper<uchar2>    VolumeUChar2;
    typedef VolumeHelper<uchar3>    VolumeUChar3;
    typedef VolumeHelper<uchar4>    VolumeUChar4;

    typedef VolumeHelper<short>     VolumeShort;
    typedef VolumeHelper<short2>    VolumeShort2;
    typedef VolumeHelper<short3>    VolumeShort3;
    typedef VolumeHelper<short4>    VolumeShort4;

    typedef VolumeHelper<uint16_t>  VolumeUShort;
    typedef VolumeHelper<ushort2>   VolumeUShort2;
    typedef VolumeHelper<ushort3>   VolumeUShort3;
    typedef VolumeHelper<ushort4>   VolumeUShort4;

    typedef VolumeHelper<int>       VolumeInt;
    typedef VolumeHelper<int2>      VolumeInt2;
    typedef VolumeHelper<int3>      VolumeInt3;
    typedef VolumeHelper<int4>      VolumeInt4;

    typedef VolumeHelper<uint32_t>  VolumeUInt;
    typedef VolumeHelper<uint2>     VolumeUInt2;
    typedef VolumeHelper<uint3>     VolumeUInt3;
    typedef VolumeHelper<uint4>     VolumeUInt4;

    typedef VolumeHelper<uint16_t>  VolumeUShort;
    typedef VolumeHelper<ushort2>   VolumeUShort2;
    typedef VolumeHelper<ushort3>   VolumeUShort3;
    typedef VolumeHelper<ushort4>   VolumeUShort4;

    typedef VolumeHelper<float>     VolumeFloat;
    typedef VolumeHelper<float2>    VolumeFloat2;
    typedef VolumeHelper<float3>    VolumeFloat3;
    typedef VolumeHelper<float4>    VolumeFloat4;

    typedef VolumeHelper<double>    VolumeDouble;
    typedef VolumeHelper<double2>   VolumeDouble2;
    typedef VolumeHelper<double3>   VolumeDouble3;
    typedef VolumeHelper<double4>   VolumeDouble4;
}

#include "volume.inl"
