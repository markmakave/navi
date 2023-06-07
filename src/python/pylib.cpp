#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <map>

#include "lumina.hpp"

namespace lumina {
namespace python {

class tensor
{
public:

    using size_type = lumina::i64;

public:

    enum type
    {
        I8,
        U8,
        I16,
        U16,
        I32,
        U32,
        I64,
        U64,
        F32,
        F64
    };

    tensor()
      : _base(nullptr),
        _order(0),
        _dtype(I64)
    {}

    tensor(py::args args, py::kwargs kwargs)
    {
        _order = args.size();
        if (kwargs.contains("dtype")) {
            _dtype = py::cast<type>(kwargs["dtype"]);
        } else {
            _dtype = I64;
        }
    }

    ~tensor()
    {
        delete _base;
    }

protected:

    void* _base;
    i64   _order;
    type  _dtype;
};

PYBIND11_MODULE(lumina, m)
{
    m.doc() = "lumina python bindings";

    py::class_<tensor>(m, "tensor")
        .def(py::init<>())
        .def(py::init<py::args, py::kwargs>());
}

} // namespace python
} // namespace lumina
