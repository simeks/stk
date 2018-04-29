#include "io.h"
#include "vtk.h"

#include "stk/common/log.h"
#include "stk/volume/volume.h"

#include <algorithm>
#include <fstream>
#include <vector>

namespace stk
{
    struct VolumeReader
    {
        // Reads the specified file
        Volume (*read)(const char* filename);

        // Length of the signature in the beginning of the file
        size_t (*signature_length)();

        // Determines based on the file signature if this reader can read it
        bool (*can_read)(const char* filename, const char* signature, size_t len);
    };
    struct VolumeWriter
    {
        // Writes the volume
        void (*write)(const char* filename, const Volume& vol);
        
        // Checks if the given filename (and extension) is supported by this format
        bool (*can_write)(const char* filename);
    };

    struct IORegistry
    {
        IORegistry()
        {
            // VTK
            VolumeReader vtk_reader = {
                vtk::read,
                vtk::signature_length,
                vtk::can_read
            };
            VolumeWriter vtk_writer = {
                vtk::write,
                vtk::can_write
            };

            readers.push_back(vtk_reader);
            writers.push_back(vtk_writer);
        }
        std::vector<VolumeReader> readers;
        std::vector<VolumeWriter> writers;
    };
    namespace
    {
        IORegistry _io_registry;
    }
    
    // Attempts to find a reader for the given file.
    // If no reader is found, VolumeReader{0} is returned
    VolumeReader find_reader(const char* filename)
    {
        // To avoid having each IO module reopen the same file over and over we
        //  read the signature of the file first, then ask each module if it
        //  (given the signature) can read the specified file.

        size_t signature_len = 0;
        for (auto& r : _io_registry.readers) {
            signature_len = std::max(signature_len, r.signature_length());
        }

        std::ifstream f;
        f.open(filename, std::ios::in|std::ios::binary);
        if (!f.is_open())
            return {0};
        
        std::vector<char> signature;
        signature.resize(signature_len);

        f.read(signature.data(), signature_len);
        f.close();

        for (auto& r : _io_registry.readers) {
            if (r.can_read(filename, signature.data(), r.signature_length()))
                return r;
        }

        LOG(Error) << "No reader found for file " << filename << ", unsupported format?";

        return {0};
    }

    VolumeWriter find_writer(const char* filename)
    {
        for (auto& w : _io_registry.writers) {
            if (w.can_write(filename))
                return w;
        }
        return {0};
    }

    Volume read_volume(const char* filename)
    {
        VolumeReader r = find_reader(filename);
        if (!r.read) {
            LOG(Error) << "Failed to read file " << filename;
            return Volume();
        }
        return r.read(filename);
    }
    void write_volume(const char* filename, const Volume& vol)
    {
        VolumeWriter w = find_writer(filename);
        FATAL_IF(!w.write) << "No writer found for file " << filename;

        w.write(filename, vol);
    }
}