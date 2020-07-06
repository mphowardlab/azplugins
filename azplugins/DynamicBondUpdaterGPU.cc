// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file DynamicBondUpdaterGPU.cc
 * \brief Definition of DynamicBondUpdaterGPU
 */

#include "DynamicBondUpdaterGPU.h"
#include "DynamicBondUpdaterGPU.cuh"

namespace azplugins
{

 DynamicBondUpdaterGPU::DynamicBondUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                          std::shared_ptr<ParticleGroup> group_1,
                          std::shared_ptr<ParticleGroup> group_2,
                          const Scalar r_cut,
                          unsigned int bond_type,
                          unsigned int max_bonds_group_1,
                          unsigned int max_bonds_group_2)
        : DynamicBondUpdater(sysdef, group_1, group_2, r_cut, bond_type, max_bonds_group_1, max_bonds_group_2)
    {
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "dynamic_bond_updater", m_exec_conf));
    }

DynamicBondUpdaterGPU::~DynamicBondUpdaterGPU()
    {
    }

namespace detail
{
/*!
 * \param m Python module to export to
 */
 void export_DynamicBondUpdaterGPU(pybind11::module& m)
     {
     namespace py = pybind11;
     py::class_< DynamicBondUpdaterGPU, std::shared_ptr<DynamicBondUpdaterGPU> >(m, "DynamicBondUpdaterGPU", py::base<DynamicBondUpdater>())
         .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>,
              std::shared_ptr<ParticleGroup>, const Scalar, unsigned int, unsigned int, unsigned int>());

         //.def_property("inside", &DynamicBondUpdater::getInsideType, &DynamicBondUpdater::setInsideType)
         //.def_property("outside", &DynamicBondUpdater::getOutsideType, &DynamicBondUpdater::setOutsideType)
         //.def_property("lo", &DynamicBondUpdater::getRegionLo, &DynamicBondUpdater::setRegionLo)
         //.def_property("hi", &DynamicBondUpdater::getRegionHi, &DynamicBondUpdater::setRegionHi);
     }
} // end namespace detail
 
} // end namespace azplugins
