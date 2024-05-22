// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

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
    .def(py::init<Scalar,Scalar3,Scalar3,bool>())
    .def_readwrite("radius", &CylinderWall::r);
    }

void export_SphereWall(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<SphereWall, std::shared_ptr<SphereWall>>(m, "_SphereWall")
    .def(py::init<Scalar,Scalar3,bool>())
    .def_readwrite("radius", &SphereWall::r);
    }
} // end namespace detail
} // end namespace azplugins

