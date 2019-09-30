#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stk/common/error.h>

#include <cassert>
#include <map>
#include <string>

namespace py = pybind11;

class VolumeW

py::array test()
{
    printf("test\n");
    return py::array_t<double>({32,32,32});
}

// Volume read_volume(const std::string& filename);

// // Attempts to write the given volume to the specified file.
// // Uses the extension of the filename to identify the target file format.
// // Triggers a fatal error if write failed (e.g. invalid file extension).
// void write_volume(const std::string&, const Volume& vol);

PYBIND11_MODULE(_stk, m)
{
    m.def("test", &test, "");
    // m.def("read_volume", &read_volume, "");
    // m.def("write_volume", &write_volume, "");
    
    // Translate relevant exception types. The exceptions not handled
    // here will be translated autmatically according to pybind11's rules.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        }
        catch (const stk::FatalException &e) {
            // Map stk::FatalException to Python RutimeError
            // +13 to remove the "Fatal error: " from the message
            PyErr_SetString(PyExc_RuntimeError, e.what() + 13);
        }
    });
}

