#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stk/common/error.h>
#include <stk/image/volume.h>

#include <cassert>
#include <map>
#include <string>

namespace py = pybind11;


/*!
 * \brief Get the stk::Type associated to a numpy image.
 */
stk::Type get_stk_type(const py::array& a) {
    stk::Type base_type = stk::Type_Unknown;

    if (py::isinstance<py::array_t<char>>(a)) {
        base_type = stk::Type_Char;
    }
    else if (py::isinstance<py::array_t<bool>>(a)) {
        base_type = stk::Type_Char;
    }
    else if (py::isinstance<py::array_t<uint8_t>>(a)) {
        base_type = stk::Type_UChar;
    }
    else if (py::isinstance<py::array_t<short>>(a)) {
        base_type = stk::Type_Short;
    }
    else if (py::isinstance<py::array_t<uint16_t>>(a)) {
        base_type = stk::Type_UShort;
    }
    else if (py::isinstance<py::array_t<int>>(a)) {
        base_type = stk::Type_Int;
    }
    else if (py::isinstance<py::array_t<uint32_t>>(a)) {
        base_type = stk::Type_UInt;
    }
    else if (py::isinstance<py::array_t<float>>(a)) {
        base_type = stk::Type_Float;
    }
    else if (py::isinstance<py::array_t<double>>(a)) {
        base_type = stk::Type_Double;
    }
    else {
        throw std::invalid_argument("Unsupported type");
    }

    // NOTE: the value of ndim can be ambiguous, e.g.
    // ndim == 3 may be a scalar volume or a vector 2D image...
    return stk::build_type(base_type, a.ndim() == 4 ? 3 : 1);
}

void get_numpy_type_and_shape(stk::Type type, const dim3& size) {
    stk::Type base_type = stk::base_type(type);

    stk::Type base_type = stk::Type_Unknown;

    if (py::isinstance<py::array_t<char>>(a)) {
        base_type = stk::Type_Char;
    }
    else if (py::isinstance<py::array_t<bool>>(a)) {
        base_type = stk::Type_Char;
    }
    else if (py::isinstance<py::array_t<uint8_t>>(a)) {
        base_type = stk::Type_UChar;
    }
    else if (py::isinstance<py::array_t<short>>(a)) {
        base_type = stk::Type_Short;
    }
    else if (py::isinstance<py::array_t<uint16_t>>(a)) {
        base_type = stk::Type_UShort;
    }
    else if (py::isinstance<py::array_t<int>>(a)) {
        base_type = stk::Type_Int;
    }
    else if (py::isinstance<py::array_t<uint32_t>>(a)) {
        base_type = stk::Type_UInt;
    }
    else if (py::isinstance<py::array_t<float>>(a)) {
        base_type = stk::Type_Float;
    }
    else if (py::isinstance<py::array_t<double>>(a)) {
        base_type = stk::Type_Double;
    }
    else {
        throw std::invalid_argument("Unsupported type");
    }

    // NOTE: the value of ndim can be ambiguous, e.g.
    // ndim == 3 may be a scalar volume or a vector 2D image...
    return stk::build_type(base_type, a.ndim() == 4 ? 3 : 1);
}


/*!
 * \brief Convert a numpy array to a stk::Volume.
 *
 * The new volume creates a copy of the input
 * image data.
 *
 * @note The numpy array must be C-contiguous, with
 *       [z,y,x] indexing.
 *
 * @param image Array representing a volume image.
 * @param origin Vector of length 3, containing
 *               the (x, y,z) coordinates of the
 *               volume origin.
 * @param spacing Vector of length 3, containing
 *                the (x, y,z) spacing of the
 *                volume.
 * @param direction Vector of length 9, representing
 *                  the cosine direction matrix in
 *                  row-major order.
 * @return A volume representing the same image.
 */
stk::Volume image_to_volume(
        const py::array image,
        const std::vector<double>& origin,
        const std::vector<double>& spacing,
        const std::vector<double>& direction
        )
{
    if (image.flags() & py::array::f_style) {
        throw std::invalid_argument("The arrays must be C-contiguous.");
    }
    
    float3 origin_ {
        float(origin[0]),
        float(origin[1]),
        float(origin[2]),
    };
    float3 spacing_ {
        float(spacing[0]),
        float(spacing[1]),
        float(spacing[2]),
    };
    Matrix3x3f direction_ {{
        {float(direction[0]), float(direction[1]), float(direction[2])},
        {float(direction[3]), float(direction[4]), float(direction[5])},
        {float(direction[6]), float(direction[7]), float(direction[8])},
    }};
    dim3 size {
        std::uint32_t(image.shape(2)),
        std::uint32_t(image.shape(1)),
        std::uint32_t(image.shape(0)),
    };
    stk::Volume volume {size, get_stk_type(image), image.data()};
    volume.set_origin(origin_);
    volume.set_spacing(spacing_);
    volume.set_direction(direction_);
    return volume;
}


class PyVolume
{
public:
    PyVolume() {}
    ~PyVolume() {}

    py::tuple origin() {
        return py::make_tuple(
            _volume.origin().x,
            _volume.origin().y,
            _volume.origin().z
        );
    }

    bool valid() {
        return _volume.valid();
    }

    void set_origin(py::tuple origin) {
        if (origin.size() != 3) {
            throw py::value_error("Expected an 3-tuple");
        }

        _volume.set_origin(float3{
            origin[0].cast<float>(),
            origin[1].cast<float>(),
            origin[2].cast<float>()
        });
    }

    py::tuple spacing() {
        return py::make_tuple(
            _volume.spacing().x,
            _volume.spacing().y,
            _volume.spacing().z
        );
    }
    void set_spacing(py::tuple spacing) {
        if (spacing.size() != 3) {
            throw py::value_error("Expected an 3-tuple");
        }

        _volume.set_spacing(float3{
            spacing[0].cast<float>(),
            spacing[1].cast<float>(),
            spacing[2].cast<float>()
        });
    }

    py::array_t<float> direction() {
        return py::array_t<float>(
            {3, 3},
            &_volume.direction()._rows[0].x
        );
    }
    void set_direction(py::array_t<float> dir) {
        if (dir.ndim() != 2
            || dir.shape(0) != 3
            || dir.shape(1) != 3) {
            throw py::value_error("Expected an 3x3 matrix");
        }

        _volume.set_direction(Matrix3x3f{
            float3{dir.at(0, 0), dir.at(0, 1), dir.at(0, 2)},
            float3{dir.at(1, 0), dir.at(1, 1), dir.at(1, 2)},
            float3{dir.at(2, 0), dir.at(2, 1), dir.at(2, 2)}
        });
    }

    py::array data() {

    }

private:
    stk::Volume _volume;
};

PyVolume test()
{
    printf("test\n");
    return PyVolume();
}

// Volume read_volume(const std::string& filename);

// // Attempts to write the given volume to the specified file.
// // Uses the extension of the filename to identify the target file format.
// // Triggers a fatal error if write failed (e.g. invalid file extension).
// void write_volume(const std::string&, const Volume& vol);

PYBIND11_MODULE(_stk, m)
{
    m.def("test", &test, "");
    
    py::class_<PyVolume>(m, "Volume")
        .def(py::init<>())
        .def("valid", &PyVolume::valid)
        .def_property("origin", &PyVolume::origin, &PyVolume::set_origin)
        .def_property("spacing", &PyVolume::spacing, &PyVolume::set_spacing)
        .def_property("direction", &PyVolume::direction, &PyVolume::set_direction)
        .def_property_readonly("data", &PyVolume::data);

        // .def("go_for_a_walk", &Pet::go_for_a_walk)
        // .def("get_hunger", &Pet::get_hunger)
        // .def("get_name", &Pet::get_name)
        // .def_property_readonly("hunger", &Pet::get_hunger)
        // .def_;

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

