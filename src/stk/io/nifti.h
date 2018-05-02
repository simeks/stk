#pragma once

#include "io.h"

// Module for nifti IO
// TODO: Support for setting/getting QForm/SForm matrices.
//       Currently both are ignored and we only read origin and spacing.

namespace stk
{
    namespace nifti
    {
        // Reading

        // Reads the specified file
        Volume read(const char* filename);

        // Length of the signature in the beginning of the file
        size_t signature_length();

        // Determines based on the file signature if this reader can read it
        bool can_read(const char* filename, const char* signature, size_t len);

        // Writing

        // Writes the given volume to the specified file
        void write(const char* filename, const Volume& vol);

        // Determines if the writer supports the given filename based on the
        //  file extension.
        bool can_write(const char* filename);
    }
} // namespace stk

