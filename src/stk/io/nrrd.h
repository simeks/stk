#pragma once

#include "io.h"

// Module for nrrd IO
// As described in http://teem.sourceforge.net/nrrd/descformat.html

namespace stk
{
    namespace nrrd
    {
        // Reading

        // Reads the specified file
        Volume read(const std::string& filename);

        // Length of the signature in the beginning of the file
        size_t signature_length();

        // Determines based on the file signature if this reader can read it
        bool can_read(const std::string& filename, const char* signature, size_t len);

        // Writing

        // Writes the given volume to the specified file
        void write(const std::string& filename, const Volume& vol);

        // Determines if the writer supports the given filename based on the
        //  file extension.
        bool can_write(const std::string& filename);
    }
} // namespace stk

