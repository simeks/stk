#pragma once

#include <string>

namespace stk
{
    class Volume;

    // Attempts to read the file with the given path and returns the loaded
    //  volume.
    // File type and IO module is automatically identified. If the file
    //  could not be found, or the format is unsupported a invalid volume
    //  (Volume::valid() == false) is returned.
    Volume read_volume(const std::string& filename);

    // Attempts to write the given volume to the specified file.
    // Uses the extension of the filename to identify the target file format.
    // Triggers a fatal error if write failed (e.g. invalid file extension).
    void write_volume(const std::string&, const Volume& vol);
}
