#include "nifti.h"
#include "stk/common/log.h"
#include "stk/image/volume.h"

#include <nifti1_io.h>

// TODO: Byte swapping?

/** Transpose vector data to have the vector index in the slowest axis */
template<typename T>
void to_slowest(
        const T * const __restrict in,
        T * const __restrict out,
        const dim3 size,
        const int nt
        )
{
    const int nx = (int) size.x;
    const int ny = (int) size.y;
    const int nz = (int) size.z;

    #pragma omp parallel for
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                for (int t = 0; t < nt; ++t) {
                    out[t*nz*ny*nx + z*ny*nx + y*nx + x] = in[z*ny*nx*nt + y*nx*nt + x*nt + t];
                }
            }
        }
    }
}

/** Transpose vector data to have the vector index in the fastest axis */
template<typename T>
void to_fastest(
        const T * const __restrict in,
        T * const __restrict out,
        const dim3 size,
        const int nt
        )
{
    const int nx = (int) size.x;
    const int ny = (int) size.y;
    const int nz = (int) size.z;

    #pragma omp parallel for
    for (int t = 0; t < nt; ++t) {
        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    out[z*ny*nx*nt + y*nx*nt + x*nt + t] = in[t*nz*ny*nx + z*ny*nx + y*nx + x];
                }
            }
        }
    }
}

namespace stk {
namespace nifti {
    void initialize()
    {
        nifti_set_debug_level(0);
    }
    Volume read(const std::string& filename)
    {
        // NIFTI-1
        // TODO: [nifti] Handle image meta data

        znzFile fp = znzopen(filename.c_str(), "rb", nifti_is_gzfile(filename.c_str()));
        if (znz_isnull(fp)) {
            LOG(Error) << "Failed to open file " << filename << " for reading";
            return Volume();
        }

        nifti_1_header nhdr = {};
        size_t r = znzread(&nhdr, 1, sizeof(nhdr), fp);
        if (r < sizeof(nhdr)) {
            LOG(Error) << "Failed to read header from file " << filename;
            znzclose(fp);
            return Volume();
        }

        int nifti_v = NIFTI_VERSION(nhdr);
        if (nifti_v != 1 || !NIFTI_ONEFILE(nhdr)) {
            LOG(Error) << "Failed to read " << filename
                       << ": Only supports NIFTI-1 and single file images";
            znzclose(fp);
            return Volume();
        }

        bool need_swap =  NIFTI_NEEDS_SWAP(nhdr);
        if (need_swap) {
            swap_nifti_header(&nhdr, nifti_v);
        }

        if (!(nhdr.dim[0] == 3
        || (nhdr.intent_code == NIFTI_INTENT_VECTOR && nhdr.dim[0] == 5)
        || (nhdr.dim[0] == 4 && nhdr.dim[4] == 1))) { // Special-case
            LOG(Error) << "Failed to read " << filename
                       << ": Only three dimensional volumes are supported";

            znzclose(fp);
            return Volume();
        }

        dim3 size = {(uint32_t)nhdr.dim[1], (uint32_t)nhdr.dim[2], (uint32_t)nhdr.dim[3]};
        const int ncomp = nhdr.intent_code == NIFTI_INTENT_VECTOR ? nhdr.dim[5] : 1;

        Type voxel_type = Type_Unknown;
        switch(nhdr.datatype)
        {
        case NIFTI_TYPE_INT8:
                 if (ncomp == 1) voxel_type = Type_Char;
            else if (ncomp == 2) voxel_type = Type_Char2;
            else if (ncomp == 3) voxel_type = Type_Char3;
            else if (ncomp == 4) voxel_type = Type_Char4;
            break;
        case NIFTI_TYPE_UINT8:
                 if (ncomp == 1) voxel_type = Type_UChar;
            else if (ncomp == 2) voxel_type = Type_UChar2;
            else if (ncomp == 3) voxel_type = Type_UChar3;
            else if (ncomp == 4) voxel_type = Type_UChar4;
            break;
        case NIFTI_TYPE_INT16:
                 if (ncomp == 1) voxel_type = Type_Short;
            else if (ncomp == 2) voxel_type = Type_Short2;
            else if (ncomp == 3) voxel_type = Type_Short3;
            else if (ncomp == 4) voxel_type = Type_Short4;
            break;
        case NIFTI_TYPE_UINT16:
                 if (ncomp == 1) voxel_type = Type_UShort;
            else if (ncomp == 2) voxel_type = Type_UShort2;
            else if (ncomp == 3) voxel_type = Type_UShort3;
            else if (ncomp == 4) voxel_type = Type_UShort4;
            break;
        case NIFTI_TYPE_INT32:
                 if (ncomp == 1) voxel_type = Type_Int;
            else if (ncomp == 2) voxel_type = Type_Int2;
            else if (ncomp == 3) voxel_type = Type_Int3;
            else if (ncomp == 4) voxel_type = Type_Int4;
            break;
        case NIFTI_TYPE_UINT32:
                 if (ncomp == 1) voxel_type = Type_UInt;
            else if (ncomp == 2) voxel_type = Type_UInt2;
            else if (ncomp == 3) voxel_type = Type_UInt3;
            else if (ncomp == 4) voxel_type = Type_UInt4;
            break;
        case NIFTI_TYPE_FLOAT32:
                 if (ncomp == 1) voxel_type = Type_Float;
            else if (ncomp == 2) voxel_type = Type_Float2;
            else if (ncomp == 3) voxel_type = Type_Float3;
            else if (ncomp == 4) voxel_type = Type_Float4;
            break;
        case NIFTI_TYPE_FLOAT64:
                 if (ncomp == 1) voxel_type = Type_Double;
            else if (ncomp == 2) voxel_type = Type_Double2;
            else if (ncomp == 3) voxel_type = Type_Double3;
            else if (ncomp == 4) voxel_type = Type_Double4;
            break;
        };

        if (voxel_type == Type_Unknown) {
            LOG(Error) << "Failed to read " << filename
                       << ": Unsupported data type (" << nhdr.datatype << ")";
            znzclose(fp);
            return Volume();
        }

        mat44 mat = nifti_quatern_to_mat44(
            nhdr.quatern_b, nhdr.quatern_c, nhdr.quatern_d,
            nhdr.qoffset_x, nhdr.qoffset_y, nhdr.qoffset_z,
            nhdr.pixdim[1], nhdr.pixdim[2], nhdr.pixdim[3],
            nhdr.pixdim[0]
        );

        Volume vol(size, voxel_type);
        vol.set_spacing({nhdr.pixdim[1], nhdr.pixdim[2], nhdr.pixdim[3]});

        // Only supported modes for now
        if (nhdr.qform_code != NIFTI_XFORM_SCANNER_ANAT) {
            LOG(Error) << "Failed to read " << filename
                       << ": Unsupported qform code, only supports NIFTI_XFORM_SCANNER_ANAT";
            znzclose(fp);
            return Volume();
        }
        if (nhdr.sform_code != NIFTI_XFORM_UNKNOWN) {
            LOG(Error) << "Failed to read " << filename
                       << ": Unsupported sform code, only supports NIFTI_XFORM_UNKNOWN";
            znzclose(fp);
            return Volume();
        }

        // Set according to method 2 (nifti1.h)

        vol.set_origin({
            -mat.m[0][3],
            -mat.m[1][3],
             mat.m[2][3]
        });

        vol.set_direction({{
            {-mat.m[0][0], -mat.m[0][1], -mat.m[0][2]},
            {-mat.m[1][0], -mat.m[1][1], -mat.m[1][2]},
            { mat.m[2][0],  mat.m[2][1],  mat.m[2][2]},
        }});

        // Offset to image data
        int vox_offset = (int)nhdr.vox_offset;
        ASSERT((size_t)vox_offset >= sizeof(nhdr));

        znzseek(fp, vox_offset, SEEK_SET);

        const size_t num_bytes = (nhdr.bitpix/8)*nhdr.dim[1]*nhdr.dim[2]*nhdr.dim[3]*ncomp;
        std::vector<char> data;

        // Read data
        if (ncomp > 1) {
            data.resize(num_bytes);
            r = znzread(data.data(), 1, num_bytes, fp);
        }
        else {
            r = znzread(vol.ptr(), 1, num_bytes, fp);
        }

        // Check read
        if (r < num_bytes) {
            LOG(Error) << "Failed to read image data from file " << filename;
            znzclose(fp);
            return Volume();
        }

        znzclose(fp);

        // Transpose vector data if needed
        if (ncomp > 1) {
            switch(nhdr.datatype)
            {
            case NIFTI_TYPE_INT8:
                to_fastest<int8_t>((int8_t*) data.data(), (int8_t*)vol.ptr(), vol.size(), ncomp);
                break;
            case NIFTI_TYPE_UINT8:
                to_fastest<uint8_t>((uint8_t*) data.data(), (uint8_t*)vol.ptr(), vol.size(), ncomp);
                break;
            case NIFTI_TYPE_INT16:
                to_fastest<int16_t>((int16_t*) data.data(), (int16_t*)vol.ptr(), vol.size(), ncomp);
                break;
            case NIFTI_TYPE_UINT16:
                to_fastest<uint16_t>((uint16_t*) data.data(), (uint16_t*)vol.ptr(), vol.size(), ncomp);
                break;
            case NIFTI_TYPE_INT32:
                to_fastest<int32_t>((int32_t*) data.data(), (int32_t*)vol.ptr(), vol.size(), ncomp);
                break;
            case NIFTI_TYPE_UINT32:
                to_fastest<uint32_t>((uint32_t*) data.data(), (uint32_t*)vol.ptr(), vol.size(), ncomp);
                break;
            case NIFTI_TYPE_FLOAT32:
                to_fastest<float>((float*) data.data(), (float*)vol.ptr(), vol.size(), ncomp);
                break;
            case NIFTI_TYPE_FLOAT64:
                to_fastest<double>((double*) data.data(), (double*)vol.ptr(), vol.size(), ncomp);
                break;
            }
        }

        int scalar_size = (int)type_size(base_type(voxel_type));
        if (need_swap) {
            nifti_swap_Nbytes(size.x*size.y*size.z*ncomp, scalar_size, vol.ptr());
        }

        return vol;
    }

    size_t signature_length()
    {
        return 0;
    }

    bool can_read(const std::string& filename, const char*, size_t)
    {
        // TODO: Signature detection on nifti

        int r = is_nifti_file(filename.c_str());
        if (r > 0)
            return true;

        return false;
    }

    void write(const std::string& filename, const Volume& vol)
    {
        ASSERT(vol.valid());
        ASSERT(vol.voxel_type() != Type_Unknown);

        // NIFTI-1

        nifti_1_header nhdr = {};
        nhdr.sizeof_hdr = sizeof(nhdr);
        nhdr.dim_info = 0;

        dim3 size = vol.size();
        int ncomp = num_components(vol.voxel_type());

        nhdr.dim[0] = ncomp == 1 ? 3 : 5;
        nhdr.dim[1] = (short)size.x;
        nhdr.dim[2] = (short)size.y;
        nhdr.dim[3] = (short)size.z;
        nhdr.dim[4] = 1;
        nhdr.dim[5] = (short)ncomp;
        nhdr.dim[6] = 1;
        nhdr.dim[7] = 1;

        // TODO: [nifti] Handle image meta data
        // intent_p1, intent_p2, intent_p3

        nhdr.intent_code = ncomp != 1 ? NIFTI_INTENT_VECTOR : 0;

        switch(base_type(vol.voxel_type()))
        {
        case Type_Char:
            nhdr.datatype = NIFTI_TYPE_INT8;
            nhdr.bitpix = 8;
            break;
        case Type_UChar:
            nhdr.datatype = NIFTI_TYPE_UINT8;
            nhdr.bitpix = 8;
            break;
        case Type_Short:
            nhdr.datatype = NIFTI_TYPE_INT16;
            nhdr.bitpix = 16;
            break;
        case Type_UShort:
            nhdr.datatype = NIFTI_TYPE_UINT16;
            nhdr.bitpix = 16;
            break;
        case Type_Int:
            nhdr.datatype = NIFTI_TYPE_INT32;
            nhdr.bitpix = 32;
            break;
        case Type_UInt:
            nhdr.datatype = NIFTI_TYPE_UINT32;
            nhdr.bitpix = 32;
            break;
        case Type_Float:
            nhdr.datatype = NIFTI_TYPE_FLOAT32;
            nhdr.bitpix = 32;
            break;
        case Type_Double:
            nhdr.datatype = NIFTI_TYPE_FLOAT64;
            nhdr.bitpix = 64;
            break;
        default:
            FATAL() << "Unsupported format";
        };

        nhdr.slice_start = 0; // TODO: [nifti] Handle image meta data

        nhdr.vox_offset = sizeof(nhdr); // TODO: [nifti] Handle image meta data

        nhdr.scl_slope = 1.0f; // TODO: [nifti] Handle slope/inter
        nhdr.scl_inter = 0.0f; // TODO: [nifti] Handle slope/inter

        // TODO: [nifti] Handle image meta data
        nhdr.slice_end = 0;
        nhdr.slice_code = 0;
        nhdr.xyzt_units = NIFTI_UNITS_MM;
        nhdr.cal_max = 0.0f;
        nhdr.cal_min = 0.0f;
        nhdr.slice_duration = 0.0f;
        nhdr.toffset = 0.0f;

        strncpy(nhdr.descrip, "STK", 3+1); // TODO: Version #
        // nhdr.descripaux_file

        // Only supported modes for now
        nhdr.qform_code = NIFTI_XFORM_SCANNER_ANAT;
        nhdr.sform_code = NIFTI_XFORM_UNKNOWN;

        // Set according to method 2 (nifti1.h)

        const Matrix3x3f& d = vol.direction();
        mat44 mat = nifti_make_orthog_mat44(
                -d(0, 0), -d(0, 1), -d(0, 2),
                -d(1, 0), -d(1, 1), -d(1, 2),
                 d(2, 0),  d(2, 1),  d(2, 2)
                );

        float3 origin = vol.origin();

        mat.m[0][3] = -origin.x;
        mat.m[1][3] = -origin.y;
        mat.m[2][3] = origin.z;

        nifti_mat44_to_quatern(mat,
            &nhdr.quatern_b,
            &nhdr.quatern_c,
            &nhdr.quatern_d,
            &nhdr.qoffset_x,
            &nhdr.qoffset_y,
            &nhdr.qoffset_z,
            &nhdr.pixdim[1],
            &nhdr.pixdim[2],
            &nhdr.pixdim[3],
            &nhdr.pixdim[0]
        );

        const float3& spacing = vol.spacing();
        nhdr.pixdim[1] *= spacing.x;
        nhdr.pixdim[2] *= spacing.y;
        nhdr.pixdim[3] *= spacing.z;
        nhdr.pixdim[4] = 1.0f;
        nhdr.pixdim[5] = 1.0f;
        nhdr.pixdim[6] = 1.0f;
        nhdr.pixdim[7] = 1.0f;

        // srow_x, srow_y, srow_z (set to all zeros)
        // intent_name

        strncpy(nhdr.magic, "n+1", 4); // Nifti1, data in same file as header

        znzFile fp = znzopen(filename.c_str(), "wb", nifti_is_gzfile(filename.c_str()));
        FATAL_IF(znz_isnull(fp))
            << "Failed to open file " << filename << " for writing";

        // Write header
        size_t w = znzwrite(&nhdr, 1, sizeof(nhdr), fp);
        if(w < sizeof(nhdr)) {
            znzclose(fp);
            FATAL() << "Failed to write to file " << filename;
        }

        std::vector<char> data;
        const size_t num_bytes = (nhdr.bitpix/8)*nhdr.dim[1]*nhdr.dim[2]*nhdr.dim[3]*ncomp;

        // Transpose data if necessary
        if (ncomp > 1) {
            data.resize(num_bytes);
            switch(nhdr.datatype)
            {
            case NIFTI_TYPE_INT8:
                to_slowest<int8_t>((int8_t*) vol.ptr(), (int8_t*)data.data(), vol.size(), ncomp);
                break;
            case NIFTI_TYPE_UINT8:
                to_slowest<uint8_t>((uint8_t*) vol.ptr(), (uint8_t*)data.data(), vol.size(), ncomp);
                break;
            case NIFTI_TYPE_INT16:
                to_slowest<int16_t>((int16_t*) vol.ptr(), (int16_t*)data.data(), vol.size(), ncomp);
                break;
            case NIFTI_TYPE_UINT16:
                to_slowest<uint16_t>((uint16_t*) vol.ptr(), (uint16_t*)data.data(), vol.size(), ncomp);
                break;
            case NIFTI_TYPE_INT32:
                to_slowest<int32_t>((int32_t*) vol.ptr(), (int32_t*)data.data(), vol.size(), ncomp);
                break;
            case NIFTI_TYPE_UINT32:
                to_slowest<uint32_t>((uint32_t*) vol.ptr(), (uint32_t*)data.data(), vol.size(), ncomp);
                break;
            case NIFTI_TYPE_FLOAT32:
                to_slowest<float>((float*) vol.ptr(), (float*)data.data(), vol.size(), ncomp);
                break;
            case NIFTI_TYPE_FLOAT64:
                to_slowest<double>((double*) vol.ptr(), (double*)data.data(), vol.size(), ncomp);
                break;
            }
        }

        // Write image data
        w = znzwrite(ncomp > 1 ? data.data() : vol.ptr(), 1, num_bytes, fp);
        if(w < num_bytes) {
            znzclose(fp);
            FATAL() << "Failed to write to file " << filename;
        }

        znzclose(fp);
    }

    bool can_write(const std::string& filename)
    {
        return nifti_is_complete_filename(filename.c_str()) > 0;
    }

} // namespace nifti
} // namespace stk
