#pragma once

#include "io.h"

// Module for nifti IO
// Implementation is still a bit sketchy and all meta-data except origin and
//  spacing is thrown away. Will change whenver a system for handling meta-data
//  is implemented.
// TODO: Support for setting/getting QForm/SForm matrices.
//       Currently both are ignored and we only read origin and spacing.

namespace stk
{
    namespace nifti
    {
        // Initializes, mostly setting up nifti global settings
        void initialize();

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

