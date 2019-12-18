// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

#include "WallRestraintCompute.h"

namespace azplugins
{

template class WallRestraintCompute<PlaneWall>;
template class WallRestraintCompute<CylinderWall>;
template class WallRestraintCompute<SphereWall>;

namespace detail
{
void export_PlaneWall(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<PlaneWall, std::shared_ptr<PlaneWall>>(m, "_PlaneWall")
    .def(py::init<Scalar3,Scalar3,bool>());
    }

void export_CylinderWall(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<CylinderWall, std::shared_ptr<CylinderWall>>(m, "_CylinderWall")
    .def(py::init<Scalar,Scalar3,Scalar3,bool>());
    }

void export_SphereWall(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<SphereWall, std::shared_ptr<SphereWall>>(m, "_SphereWall")
    .def(py::init<Scalar,Scalar3,bool>());
    }
} // end namespace detail
} // end namespace azplugins

