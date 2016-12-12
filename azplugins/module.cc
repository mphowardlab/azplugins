#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

PYBIND11_PLUGIN(_azplugins)
    {
    pybind11::module m("_azplugins");

    return m.ptr();
    }
