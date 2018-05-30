#include "nrrd.h"

#include "stk/common/log.h"
#include "stk/image/volume.h"

#include <assert.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "NrrdIO.h"

namespace
{
    // Helper function to get last error message from NRRD without causing any
    // memory leaks
    std::string nrrd_error()
    {
        char* err = biffGetDone(NRRD);
        std::string ret = err;
        free(err);
        return ret;
    }
}

namespace stk {
namespace nrrd {
    Volume read(const std::string& filename)
    {
        Nrrd* nimg = nrrdNew();

        if (nrrdLoad(nimg, filename.c_str(), nullptr)) {
            LOG(Error) << "Failed to read '" << filename << "':\n" << nrrd_error();
            nrrdNuke(nimg);
            return Volume();
        }

        uint32_t range_axis_idx[NRRD_DIM_MAX];
        uint32_t range_axis_n = nrrdRangeAxesGet(nimg, range_axis_idx);
        
        // TODO: Vector data should be the fastest axis, otherwise we need to permute
        if (range_axis_n != 0 && range_axis_idx[0] != 0) {
            LOG(Error) << "Failed to load '" << filename 
                       << "': Vector data not fastest axis in memory, data requires permutation.";
            nrrdNuke(nimg);
            return Volume();
        }

        if ((range_axis_n == 0 && nimg->dim != 3) ||
            (range_axis_n > 0 && nimg->dim != 4)) {
            LOG(Error) << "Failed to load '" << filename 
                       << "': Only 3D data supported.";
            nrrdNuke(nimg);
            return Volume();
        }

        dim3 size = {0};
        int ncomp = 1;
        if (range_axis_n == 0) {
            size = {
                (uint32_t)nimg->axis[0].size,
                (uint32_t)nimg->axis[1].size,
                (uint32_t)nimg->axis[2].size
            };
        }
        else if (range_axis_n == 1) {
            ncomp = (int)nimg->axis[0].size;
            size = {
                (uint32_t)nimg->axis[1].size,
                (uint32_t)nimg->axis[2].size,
                (uint32_t)nimg->axis[3].size
            };
        }
        else {
            LOG(Error) << "Failed to load '" << filename 
                       << "': Unsupported number of axes.";
            nrrdNuke(nimg);
            return Volume();
        }
        
        Type voxel_type = Type_Unknown;
        
        switch (nimg->type) {
        case nrrdTypeChar:
                 if (ncomp == 1) voxel_type = Type_Char;
            else if (ncomp == 2) voxel_type = Type_Char2;
            else if (ncomp == 3) voxel_type = Type_Char3;
            else if (ncomp == 4) voxel_type = Type_Char4;
            break;
        case nrrdTypeUChar:
                 if (ncomp == 1) voxel_type = Type_UChar;
            else if (ncomp == 2) voxel_type = Type_UChar2;
            else if (ncomp == 3) voxel_type = Type_UChar3;
            else if (ncomp == 4) voxel_type = Type_UChar4;
            break;
        case nrrdTypeShort:
                 if (ncomp == 1) voxel_type = Type_Short;
            else if (ncomp == 2) voxel_type = Type_Short2;
            else if (ncomp == 3) voxel_type = Type_Short3;
            else if (ncomp == 4) voxel_type = Type_Short4;
            break;
        case nrrdTypeUShort:
                 if (ncomp == 1) voxel_type = Type_UShort;
            else if (ncomp == 2) voxel_type = Type_UShort2;
            else if (ncomp == 3) voxel_type = Type_UShort3;
            else if (ncomp == 4) voxel_type = Type_UShort4;
            break;
        case nrrdTypeInt:
                 if (ncomp == 1) voxel_type = Type_Int;
            else if (ncomp == 2) voxel_type = Type_Int2;
            else if (ncomp == 3) voxel_type = Type_Int3;
            else if (ncomp == 4) voxel_type = Type_Int4;
            break;
        case nrrdTypeUInt:
                 if (ncomp == 1) voxel_type = Type_UInt;
            else if (ncomp == 2) voxel_type = Type_UInt2;
            else if (ncomp == 3) voxel_type = Type_UInt3;
            else if (ncomp == 4) voxel_type = Type_UInt4;
            break;
        case nrrdTypeFloat:
                 if (ncomp == 1) voxel_type = Type_Float;
            else if (ncomp == 2) voxel_type = Type_Float2;
            else if (ncomp == 3) voxel_type = Type_Float3;
            else if (ncomp == 4) voxel_type = Type_Float4;
            break;
        case nrrdTypeDouble:
                 if (ncomp == 1) voxel_type = Type_Double;
            else if (ncomp == 2) voxel_type = Type_Double2;
            else if (ncomp == 3) voxel_type = Type_Double3;
            else if (ncomp == 4) voxel_type = Type_Double4;
            break;
        }

        if (voxel_type == Type_Unknown) {
            LOG(Error) << "Failed to load '" << filename 
                       << "': Unsupported format (nrrd->type=" 
                       << nimg->type << ", ncomp=" << ncomp << ").";
            nrrdNuke(nimg);
            return Volume();
        }

        // Origin
        float3 origin = {0};

        if (nimg->spaceDim) {
            double space_origin[NRRD_SPACE_DIM_MAX];
            for (uint32_t i = 0; i < nimg->spaceDim; ++i) {
                space_origin[i] = nimg->spaceOrigin[i];
            }

            // Correct according to space, should be nrrdSpaceLeftPosteriorSuperior
            switch (nimg->space) {
            case nrrdSpaceRightAnteriorSuperior:
                space_origin[0] = -space_origin[0];
                space_origin[1] = -space_origin[1];
                break;
            case nrrdSpaceLeftAnteriorSuperior:
                space_origin[0] = -space_origin[0];
                break;
            }
            origin = {
                (float)space_origin[0], 
                (float)space_origin[1], 
                (float)space_origin[2]
            };
        }
        else {
            LOG(Warning) << "nrrd: Could not extract origin from " << filename;
        }

        // Spacing
        float spacing[3] = {1, 1, 1};

        uint32_t domain_axis_idx[NRRD_DIM_MAX];
        uint32_t domain_axis_n = nrrdDomainAxesGet(nimg, domain_axis_idx);
        ASSERT(domain_axis_n == 3); // Should have been verified above

        for (int i = 0; i < 3; ++i) {
            double spacing_i;
            double spacing_dir[NRRD_SPACE_DIM_MAX];
            int status = nrrdSpacingCalculate(nimg, domain_axis_idx[i], &spacing_i, 
                spacing_dir);

            switch (status) {
            case nrrdSpacingStatusUnknown:
                LOG(Warning) << "nrrd: Could not extract spacing[" << i << "] from " << filename;
                break;
            case nrrdSpacingStatusScalarNoSpace:
            case nrrdSpacingStatusScalarWithSpace:
            case nrrdSpacingStatusDirection:
                if (AIR_EXISTS(spacing_i)) {
                    spacing[i] = (float)spacing_i;
                }
                break;
            }
        }

        Volume vol(size, voxel_type);
        vol.set_origin(origin);
        vol.set_spacing({spacing[0], spacing[1], spacing[2]});

        memcpy(vol.ptr(), nimg->data, nrrdElementSize(nimg) * nrrdElementNumber(nimg));

        // nrrd allocated memory for the volume data itself, therefore we need to use nuke
        nrrdNuke(nimg);

        return vol;
    }

    size_t signature_length()
    {
        // First line in file
        // "NRRD"
        return 4;
    }

    bool can_read(const std::string& /*filename*/, const char* signature, size_t len)
    {
        return len >= signature_length() &&
                memcmp(signature, "NRRD", 4) == 0;
    }

    void write(const std::string& file, const Volume& vol)
    {
        ASSERT(vol.valid());
        ASSERT(vol.voxel_type() != Type_Unknown);

        Nrrd* nimg = nrrdNew();

        int kind[NRRD_DIM_MAX];
        double space_dir[NRRD_DIM_MAX][NRRD_SPACE_DIM_MAX]; // 3d only
        size_t size[NRRD_DIM_MAX];

        int base = 0;

        Type voxel_type = vol.voxel_type();
        int num_comp = num_components(voxel_type);
        if (num_comp > 1) {
            kind[0] = nrrdKindVector;
            size[0] = num_comp;
            base = 1;

            space_dir[0][0] = AIR_NAN;
            space_dir[0][1] = AIR_NAN;
            space_dir[0][2] = AIR_NAN;
        }

        int nrrd_type = nrrdTypeUnknown;
        switch (base_type(voxel_type)) {
        case Type_Char:
            nrrd_type = nrrdTypeChar;
            break;
        case Type_UChar:
            nrrd_type = nrrdTypeUChar;
            break;
        case Type_Short:
            nrrd_type = nrrdTypeShort;
            break;
        case Type_UShort:
            nrrd_type = nrrdTypeUShort;
            break;
        case Type_Int:
            nrrd_type = nrrdTypeInt;
            break;
        case Type_UInt:
            nrrd_type = nrrdTypeUInt;
            break;
        case Type_Float:
            nrrd_type = nrrdTypeFloat;
            break;
        case Type_Double:
            nrrd_type = nrrdTypeDouble;
            break;
        default:
            nrrdNix(nimg);
            FATAL() << "Unsupported format";
        };

        // 3d only
        size[base  ] = vol.size().x;
        size[base+1] = vol.size().y;
        size[base+2] = vol.size().z;

        if (nrrdWrap_nva(nimg, const_cast<void*>(vol.ptr()), nrrd_type, base + 3, size)) {
            nrrdNix(nimg);
            FATAL() << "Failed to write file '" << file << "':\n" << nrrd_error();
        }
        nrrdSpaceDimensionSet(nimg, 3);
        nrrdSpaceSet(nimg, nrrdSpaceLeftPosteriorSuperior);

        double origin[] = {vol.origin().x, vol.origin().y, vol.origin().z};
        if (nrrdSpaceOriginSet(nimg, origin)) {
            nrrdNix(nimg);
            FATAL() << "Failed to write file '" << file << "':\n" << nrrd_error();
        }

        kind[base  ] = nrrdKindDomain;
        kind[base+1] = nrrdKindDomain;
        kind[base+2] = nrrdKindDomain;
        nrrdAxisInfoSet_nva(nimg, nrrdAxisInfoKind, kind);

        space_dir[base][0] = vol.spacing().x;
        space_dir[base][1] = space_dir[base][2] = 0;
        
        space_dir[base+1][1] = vol.spacing().y;
        space_dir[base+1][0] = space_dir[base+1][2] = 0;

        space_dir[base+2][2] = vol.spacing().z;
        space_dir[base+2][0] = space_dir[base+2][1] = 0;

        nrrdAxisInfoSet_nva(nimg, nrrdAxisInfoSpaceDirection, space_dir);

        NrrdIoState *nio = nrrdIoStateNew();

        // Always use gzip if available
        if (nrrdEncodingGzip->available()) {
            nio->encoding = nrrdEncodingGzip;
        }
        else {
            LOG(Info) << "Nrrd: Gzip not available.";
            nio->encoding = nrrdEncodingRaw;
        }
        nio->endian = airEndianLittle;

        if (nrrdSave(file.c_str(), nimg, nio)) {
            nrrdIoStateNix(nio);
            nrrdNix(nimg);
            FATAL() << "Failed to write file '" << file << "':\n" << nrrd_error();
        }

        nrrdIoStateNix(nio);

        // nrrdNix is prefered as nrrdNuke would free the external volume data
        nrrdNix(nimg);
    }

    bool can_write(const std::string& filename)
    {
        size_t p = filename.rfind('.');
        if (p == std::string::npos)
            return false;

        std::string ext = filename.substr(p+1); // Skip '.'
        
        for (size_t i = 0; i < ext.length(); ++i) 
            ext[i] = (char)::tolower(ext[i]);

        return ext == "nrrd";
    }

} // namespace nrrd
} // namespace stk
