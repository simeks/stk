#include "nifti.h"

#include <nifti1_io.h>

namespace nifti
{
    Volume read(const char* filename)
    {
        nifti_image* nim = nifti_image_read(filename, 1);
        if (!nim) {
            return Volume();
        }

        FATAL_IF(nim->ndim != 3) << "Nifti IO: Only supports 3 dimensional data";

        int ncomp = 1;
        if (nim->intent_code == NIFTI_INTENT_VECTOR) {
            ncomp = nim->dim[5];
        }

        uint8_t voxel_type = voxel::Type_Unknown;
        switch (nim->datatype) {
            case NIFTI_TYPE_UINT8:
                if (ncomp == 1) voxel_type = Type_UChar;
                if (ncomp == 2) voxel_type = Type_UChar2;
                if (ncomp == 3) voxel_type = Type_UChar3;
                if (ncomp == 4) voxel_type = Type_UChar4;
            break;
            case NIFTI_TYPE_FLOAT32:
                if (ncomp == 1) voxel_type = Type_Float;
                if (ncomp == 2) voxel_type = Type_Float2;
                if (ncomp == 3) voxel_type = Type_Float3;
                if (ncomp == 4) voxel_type = Type_Float4;
            break;
            case NIFTI_TYPE_FLOAT64:
                if (ncomp == 1) voxel_type = Type_Double;
                if (ncomp == 2) voxel_type = Type_Double2;
                if (ncomp == 3) voxel_type = Type_Double3;
                if (ncomp == 4) voxel_type = Type_Double4;
            break;
        }

        FATAL_IF(voxel_type == voxel::Type_Unknown) << "Nifti IO: Unrecognized data type (" << nim->datatype << ")";
        
        dim3 size {
            nim.nx,
            nim.ny,
            nim.nz
        };
        float3 spacing {
            nim.dx,
            nim.dy,
            nim.dz
        };
        float3 origin {0, 0, 0};

        // TODO: Don't really know what these mean
        ASSERT(nim->qform_code == 0 && nim->sform_code == 0);
        mat44 t = nim->sto_xyz;

        origin.x = -t.m[0][3]
        origin.y = -t.m[1][3]
        origin.z = -t.m[2][3]

        Volume vol(size, voxel_type);
        vol.set_origin(origin);
        vol.set_spacing(spacing);

        size_t num_bytes = size.x * size.y * size.z * voxel::size(voxel_type);
        ASSERT(num_bytes == nim->nvox * nim->nbyper);

        memcpy((char*)vol.ptr(), nim->data, num_bytes);

        nifti_image_free(nim);
        return out;
    }

    size_t signature_length()
    {
        return 0;
    }

    bool can_read(const char* filename, const char*, size_t)
    {
        // TODO: Signature detection on nifti
        
        int r = is_nifti_file(filename);
        if (r > 0)
            return true;

        return false;
    }

    void write(const char* filename, const Volume& vol)
    {
        nifti_image* nim = nifti_simple_init_nim();
        nim->nifti_type = NIFTI_FTYPE_NIFTI1_1;

        const char* ext = nifti_find_file_extension(filename);
        FATAL_IF(!ext) 
            << "Bad nifti file name: " << filename;

        bool is_compressed = false;
        if (strcmp(ext[strlen(ext)-3), ".gz") == 0) {
            is_compressed = true;
        }

        const char* basename = nifti_makebasename(filename);
        nim->fname = nifti_makehdrname(basename, nim->nifti_type, false, is_compressed);
        nim->iname = nifti_makeimgname(basename, nim->nifti_type, false, is_compressed);
        free(basename);

        dim3 size = vol.size();

        nim->xyz_units = NIFTI_UNITS_MM;
        nim->dim[7] = nim->nw = 1;
        nim->dim[6] = nim->nv = 1;
        nim->dim[5] = nim->nu = 1;
        nim->dim[4] = nim->nt = 1;

        nim->dim[1] = nim->nx = size.x;
        nim->dim[2] = nim->ny = size.y;
        nim->dim[3] = nim->nz = size.z;


        switch(base_type(vol.voxel_type()))
        {
        case Type_Float:
            nim->datatype = NIFTI_TYPE_FLOAT32;
            nim->nbyper = 4;
        case Type_Double:
            nim->datatype = NIFTI_TYPE_FLOAT64;
            nim->nbyper = 8;            
        case Type_UChar:
            nim->datatype = NIFTI_TYPE_UINT8;
            nim->nbyper = 1;
        };
        
        int ncomp = num_components(vol.voxel_type());
        if (ncomp == 1) {
            nim->ndim = 3;
            num->dim[0] = 3;
        }
        else {

        }

        nim->qform_code = 0;
        nim->scl_slope = 1.0f;
        nim->scl_inter = 0.0f;


        nifti_image_free(nim);
    }

    bool can_write(const char* filename)
    {
        return nifti_is_complete_filename(filename) > 0;
    }
}
