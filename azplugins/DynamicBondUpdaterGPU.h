// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file DynamicBondUpdaterGPU.h
 * \brief Declaration of DynamicBondUpdaterGPU.h
 */

#ifndef AZPLUGINS_DYNAMIC_BOND_UPDATER_GPU_H_
#define AZPLUGINS_DYNAMIC_BOND_UPDATER_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "DynamicBondUpdater.h"
#include "hoomd/Autotuner.h"

namespace azplugins
{

//! Particle type updater on the GPU
/*!
 * See DynamicBondUpdater for details. This class inherits and minimally implements
 * the CPU methods from DynamicBondUpdater on the GPU.
 */
class PYBIND11_EXPORT DynamicBondUpdaterGPU : public DynamicBondUpdater
    {
    public:
      //! Constructor with parameters
      DynamicBondUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                  std::shared_ptr<ParticleGroup> group_1,
                  std::shared_ptr<ParticleGroup> group_2,
                  const Scalar r_cut,
                  unsigned int bond_type,
                  unsigned int max_bonds_group_1,
                  unsigned int max_bonds_group_2);

      //! Destructor
      virtual ~DynamicBondUpdaterGPU();

      //! find and make new bonds
    //  virtual void update(unsigned int timestep);

    protected:
  
          virtual void filterPossibleBonds();
    private:
        std::unique_ptr<Autotuner> m_tuner_filter_bonds; //!< Tuner for existing bond filter
        GPUFlags<int> m_num_nonzero_bonds;//!< GPU flags for the number of marked particles

    };

namespace detail
{
//! Export DynamicBondUpdaterGPU to python
void export_DynamicBondUpdaterGPU(pybind11::module& m);
} // end namespace detail

} // end namespace azplugins

#endif // AZPLUGINS_DYNAMIC_BOND_UPDATER_GPU_H_
