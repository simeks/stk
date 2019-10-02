#include <pybind11/buffer_info.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stk/common/error.h>
#include <stk/filters/vector_calculus.h>
#include <stk/image/volume.h>
#include <stk/io/io.h>

#include <cassert>
#include <map>
#include <string>
#include <vector>

namespace py = pybind11;


namespace pybind11 { namespace detail {
    template <> struct type_caster<float3> {
    public:
        PYBIND11_TYPE_CASTER(float3, _("float3"));

        bool load(handle src, bool) {
            try {
                tuple t = py::cast<tuple>(src);
                if (t.size() != 3) {
                    return false;
                }
                value = float3{
                    t[0].cast<float>(),
                    t[1].cast<float>(),
                    t[2].cast<float>()
                };
            } catch (std::exception& e) {
                return false;
            }
            return true;
        }
        static handle cast(float3 src, return_value_policy /* policy */, handle /* parent */) {
            return make_tuple(
                src.x,
                src.y,
                src.z
            ).release();
        }
    };
    template <> struct type_caster<dim3> {
    public:
        PYBIND11_TYPE_CASTER(dim3, _("dim3"));

        bool load(handle src, bool) {
            try {
                tuple t = py::cast<tuple>(src);
                if (t.size() != 3) {
                    return false;
                }
                value = dim3{
                    t[0].cast<uint32_t>(),
                    t[1].cast<uint32_t>(),
                    t[2].cast<uint32_t>()
                };
            } catch (std::exception& e) {
                return false;
            }
            return true;
        }
        static handle cast(dim3 src, return_value_policy /* policy */, handle /* parent */) {
            return make_tuple(
                src.x,
                src.y,
                src.z
            ).release();
        }
    };
    template <> struct type_caster<int3> {
    public:
        PYBIND11_TYPE_CASTER(int3, _("int3"));

        bool load(handle src, bool) {
            try {
                tuple t = py::cast<tuple>(src);
                if (t.size() != 3) {
                    return false;
                }
                value = int3{
                    t[0].cast<int>(),
                    t[1].cast<int>(),
                    t[2].cast<int>()
                };
            } catch (std::exception& e) {
                return false;
            }
            return true;
        }
        static handle cast(int3 src, return_value_policy /* policy */, handle /* parent */) {
            return make_tuple(
                src.x,
                src.y,
                src.z
            ).release();
        }
    };
    template <> struct type_caster<Matrix3x3f> {
    public:
        PYBIND11_TYPE_CASTER(Matrix3x3f, _("Matrix3x3f"));

        bool load(handle src, bool) {
            try {
                array_t<float> a = py::cast<array_t<float>>(src);
                if (a.ndim() != 2
                    || a.shape(0) != 3
                    || a.shape(1) != 3) {
                    return false;
                }
                value = Matrix3x3f{
                    float3{a.at(0, 0), a.at(0, 1), a.at(0, 2)},
                    float3{a.at(1, 0), a.at(1, 1), a.at(1, 2)},
                    float3{a.at(2, 0), a.at(2, 1), a.at(2, 2)}
                };
            } catch (std::exception& e) {
                return false;
            }
            return true;
        }
        static handle cast(Matrix3x3f src, return_value_policy /* policy */, handle /* parent */) {
            return py::array_t<float>(
                {3, 3},
                &src._rows[0].x
            ).release();
        }
    };

}} // namespace pybind11::detail

std::vector<size_t> get_shape(const stk::Volume& vol)
{
    dim3 size = vol.size();
    size_t ncomp = stk::num_components(vol.voxel_type());
    if (ncomp == 1) {
        // Flip directions since size is in (width, height, depth),
        // numpy expects (depth, height, width)
        return {
            size.z,
            size.y,
            size.x
        };
    }
    else {
        return {
            size.z,
            size.y,
            size.x,
            ncomp
        };
    }
}

std::vector<size_t> get_strides(const stk::Volume& vol)
{
    const size_t* strides = vol.strides();
    size_t ncomp = stk::num_components(vol.voxel_type());

    // Reverse, because of numpy ordering
    if (ncomp == 1) {
        return {
            strides[2],
            strides[1],
            strides[0]
        };
    }
    else {
        return {
            strides[2],
            strides[1],
            strides[0],
            strides[0]/ncomp
        };
    }
}

const char* format_descriptor(stk::Type base_type)
{
         if (base_type == stk::Type_Char)   return "b";
    else if (base_type == stk::Type_UChar)  return "B";
    else if (base_type == stk::Type_Short)  return "h";
    else if (base_type == stk::Type_UShort) return "H";
    else if (base_type == stk::Type_Int)    return "i";
    else if (base_type == stk::Type_UInt)   return "I";
    else if (base_type == stk::Type_Float)  return "f";
    else if (base_type == stk::Type_Double) return "d";
    return "";
}

py::buffer_info get_buffer_info(stk::Volume& v)
{
    stk::Type base_type = stk::base_type(v.voxel_type());
    int ncomp = stk::num_components(v.voxel_type());

    return py::buffer_info(
        v.ptr(),                        /* Pointer to buffer */
        stk::type_size(base_type),      /* Size of one scalar */
        format_descriptor(base_type),   /* Python struct-style format descriptor */
        ncomp == 1 ? 3 : 4,             /* Number of dimensions */
        
        get_shape(v),                   /* Buffer dimensions */
        get_strides(v)                  /* Strides (in bytes) for each index */
    );
}

stk::Volume numpy_to_volume(const py::array arr)
{
    // Ensure ordering
    py::array buf = py::array::ensure(arr, py::array::c_style | py::array::forcecast);
   
    if (!(buf.ndim() == 3 || buf.ndim() == 4)) {
        throw std::invalid_argument("Expected a 3D or a 4D array");
    }

    dim3 size {
        std::uint32_t(buf.shape(2)),
        std::uint32_t(buf.shape(1)),
        std::uint32_t(buf.shape(0)),
    };

    stk::Type base_type = stk::Type_Unknown;
    if (py::isinstance<py::array_t<char>>(arr)) {
        base_type = stk::Type_Char;
    }
    else if (py::isinstance<py::array_t<bool>>(arr)) {
        base_type = stk::Type_Char;
    }
    else if (py::isinstance<py::array_t<uint8_t>>(arr)) {
        base_type = stk::Type_UChar;
    }
    else if (py::isinstance<py::array_t<short>>(arr)) {
        base_type = stk::Type_Short;
    }
    else if (py::isinstance<py::array_t<uint16_t>>(arr)) {
        base_type = stk::Type_UShort;
    }
    else if (py::isinstance<py::array_t<int>>(arr)) {
        base_type = stk::Type_Int;
    }
    else if (py::isinstance<py::array_t<uint32_t>>(arr)) {
        base_type = stk::Type_UInt;
    }
    else if (py::isinstance<py::array_t<float>>(arr)) {
        base_type = stk::Type_Float;
    }
    else if (py::isinstance<py::array_t<double>>(arr)) {
        base_type = stk::Type_Double;
    }
    else {
        throw std::invalid_argument("Unsupported type");
    }

    int ncomp = 1;
    if (buf.ndim() == 4) {
        ncomp = buf.shape(3);
    }

    if (ncomp < 1 || ncomp > 4) {
        throw std::invalid_argument("Unsupported number of channels");
    }

    // NOTE: the value of ndim can be ambiguous, e.g.
    // ndim == 3 may be a scalar volume or a vector 2D image...

    return stk::Volume(
        size,
        stk::build_type(base_type, ncomp),
        buf.data()
    );
}

stk::Volume make_volume(
    py::array arr,
    const float3& origin,
    const float3& spacing,
    const Matrix3x3f& direction
) {
    stk::Volume vol = numpy_to_volume(arr);
    vol.set_origin(origin);
    vol.set_spacing(spacing);
    vol.set_direction(direction);
    return vol;
}


std::string divergence_docstring =
R"(Compute the divergence of a displacement field.

The divergence of a vector field

.. math::
    f(\boldsymbol{x}) =
        (f_1(\boldsymbol{x}),
         f_2(\boldsymbol{x}),
         f_3(\boldsymbol{x}))

with :math:`\boldsymbol{x} = (x_1, x_2, x_3)`
is defined as

.. math::
    \nabla \cdot f (\boldsymbol{x}) =
    \sum_{i=1}^3
        \frac{\partial f_i}{\partial x_i} (\boldsymbol{x})

.. note::
    All the arrays must be C-contiguous.

Parameters
----------
displacement: np.ndarray
    Displacement field used to resample the image.

origin: np.ndarray
    Origin of the displacement field.

spacing: np.ndarray
    Spacing of the displacement field.

direction: Tuple[Int]
    Cosine direction matrix of the displacement field.

Returns
-------
np.ndarray
    Scalar volume image containing the divergence of
    the input displacement.
)";



std::string rotor_docstring =
R"(Compute the rotor of a displacement field.

The rotor of a 3D 3-vector field

.. math::
    f(\boldsymbol{x}) =
        (f_1(\boldsymbol{x}),
         f_2(\boldsymbol{x}),
         f_3(\boldsymbol{x}))

with :math:`\boldsymbol{x} = (x_1, x_2, x_3)`
is defined as

.. math::
    \nabla \times f(\boldsymbol{x}) =
    \left(
        \frac{\partial f_3}{\partial x_2} -
        \frac{\partial f_2}{\partial x_3},
        \frac{\partial f_1}{\partial x_3} -
        \frac{\partial f_3}{\partial x_1},
        \frac{\partial f_2}{\partial x_1} -
        \frac{\partial f_1}{\partial x_2}
    \right)

.. note::
    All the arrays must be C-contiguous.

Parameters
----------
displacement: stk.Volume
    Displacement field used to resample the image.

Returns
-------
stk.Volume
    Vector volume image containing the rotor of
    the input displacement.
)";



std::string circulation_density_docstring =
R"(Compute the circulation density of a displacement field.

The circulation density for a 3D 3-vector field is defined as the
norm of the rotor.

.. note::
    All the arrays must be C-contiguous.

Parameters
----------
displacement: stk.Volume
    Displacement field used to resample the image.

Returns
-------
stk.Volume
    Vector volume image containing the circulation
    density of the input displacement.
)";

PYBIND11_MODULE(_stk, m)
{
    py::enum_<stk::Type>(m, "Type")
        .value("Unknown", stk::Type_Unknown)
        .value("Char", stk::Type_Char)
        .value("Char2", stk::Type_Char2)
        .value("Char3", stk::Type_Char3)
        .value("Char4", stk::Type_Char4)
        .value("UChar", stk::Type_UChar)
        .value("UChar2", stk::Type_UChar2)
        .value("UChar3", stk::Type_UChar3)
        .value("UChar4", stk::Type_UChar4)
        .value("Short", stk::Type_Short)
        .value("Short2", stk::Type_Short2)
        .value("Short3", stk::Type_Short3)
        .value("Short4", stk::Type_Short4)
        .value("UShort", stk::Type_UShort)
        .value("UShort2", stk::Type_UShort2)
        .value("UShort3", stk::Type_UShort3)
        .value("UShort4", stk::Type_UShort4)
        .value("Int", stk::Type_Int)
        .value("Int2", stk::Type_Int2)
        .value("Int3", stk::Type_Int3)
        .value("Int4", stk::Type_Int4)
        .value("UInt", stk::Type_UInt)
        .value("UInt2", stk::Type_UInt2)
        .value("UInt3", stk::Type_UInt3)
        .value("UInt4", stk::Type_UInt4)
        .value("Float", stk::Type_Float)
        .value("Float2", stk::Type_Float2)
        .value("Float3", stk::Type_Float3)
        .value("Float4", stk::Type_Float4)
        .value("Double", stk::Type_Double)
        .value("Double2", stk::Type_Double2)
        .value("Double3", stk::Type_Double3)
        .value("Double4", stk::Type_Double4)
        .export_values();

    py::class_<stk::Volume>(m, "Volume", py::buffer_protocol())
        .def_buffer(get_buffer_info)
        .def(py::init<>())
        .def(py::init(&make_volume),
            py::arg("array"),
            py::arg("origin") = float3{0.0, 0.0, 0.0},
            py::arg("spacing") = float3{1.0, 1.0, 1.0},
            py::arg("direction") = Matrix3x3f::Identity
        )
        .def("valid", &stk::Volume::valid)
        .def("copy_meta_from", &stk::Volume::copy_meta_from)
        .def_property_readonly("size", &stk::Volume::size)
        .def_property_readonly("type", &stk::Volume::voxel_type)
        .def_property("origin", &stk::Volume::origin, &stk::Volume::set_origin)
        .def_property("spacing", &stk::Volume::spacing, &stk::Volume::set_spacing)
        .def_property("direction", &stk::Volume::direction, &stk::Volume::set_direction)
        ;

    py::implicitly_convertible<py::array, stk::Volume>();

    m.def("read_volume", &stk::read_volume, "");
    m.def("write_volume", &stk::write_volume, "");
    
    m.def("divergence",
            &stk::divergence<float3>,
            divergence_docstring.c_str()
         );

    m.def("rotor",
            &stk::rotor<float3>,
            rotor_docstring.c_str()
         );

    m.def("circulation_density",
            &stk::circulation_density<float3>,
            circulation_density_docstring.c_str()
         );

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

