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
        : DynamicBondUpdater(sysdef, group_1, group_2, r_cut, bond_type, max_bonds_group_1, max_bonds_group_2),
        m_num_nonzero_bonds(m_exec_conf)
    {
    m_tuner_filter_bonds.reset(new Autotuner(32, 1024, 32, 5, 100000, "dynamic_bond_updater_filter_bonds", m_exec_conf));

    }

DynamicBondUpdaterGPU::~DynamicBondUpdaterGPU()
    {
    }

/*void DynamicBondUpdaterGPU::findAllPossibleBonds()
    {

    }
*/

void DynamicBondUpdaterGPU::filterPossibleBonds()
{

  //todo: figure out in which order the thrust calls are the fastest.
  // suspect: sort - remove zeros - unique - filter -remove zeros ?

  const unsigned int size = m_group_2->getNumMembers()*m_max_bonds;

  // sort and remove all existing zeros
  ArrayHandle<unsigned int> d_n_existing_bonds(m_n_existing_bonds, access_location::device, access_mode::read);
  ArrayHandle<unsigned int> d_existing_bonds_list(m_existing_bonds_list, access_location::device, access_mode::read);
  ArrayHandle<Scalar3> d_all_possible_bonds(m_all_possible_bonds, access_location::device, access_mode::readwrite);

  gpu::sort_and_remove_zeros_possible_bond_array(d_all_possible_bonds.data,
                                       size,
                                       m_num_nonzero_bonds.getDeviceFlags());


  m_num_all_possible_bonds = m_num_nonzero_bonds.readFlags();

  //filter out the existing bonds
  m_tuner_filter_bonds->begin();
  gpu::filter_existing_bonds(d_all_possible_bonds.data,
                             d_n_existing_bonds.data,
                             d_existing_bonds_list.data,
                             m_existing_bonds_list_indexer,
                             m_num_all_possible_bonds,
                             m_tuner_filter_bonds->getParam());
  m_tuner_filter_bonds->end();


  // filtering existing bonds out introduced some zeros back into the array, remove them
  gpu::remove_zeros_possible_bond_array(d_all_possible_bonds.data,
                                       m_num_all_possible_bonds,
                                       m_num_nonzero_bonds.getDeviceFlags());


  m_num_all_possible_bonds = m_num_nonzero_bonds.readFlags();

// at this point, the sub-array: h_all_possible_bonds[0,m_num_all_possible_bonds]
// should contain only unique entries of possible bonds which are not yet formed.
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
