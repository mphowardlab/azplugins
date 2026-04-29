// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
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
#include "DynamicBondUpdaterGPU.cuh"
#include "hoomd/Autotuner.h"

#include "hip/hip_runtime.h"
#include "hoomd/md/NeighborListGPUTree.cuh"


namespace hoomd
{
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
      //! Simple constructor
      DynamicBondUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<Trigger> trigger,
                         std::shared_ptr<md::NeighborList> pair_nlist,
                         std::shared_ptr<ParticleGroup> group_1,
                         std::shared_ptr<ParticleGroup> group_2,
                         uint16_t seed);


      //! Constructor with parameters
      DynamicBondUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<Trigger> trigger,
                         std::shared_ptr<md::NeighborList> pair_nlist,
                         std::shared_ptr<ParticleGroup> group_1,
                         std::shared_ptr<ParticleGroup> group_2,
                         uint16_t seed,
                         const Scalar r_cut,
                         const Scalar probability,
                         unsigned int max_bonds_group_1,
                         unsigned int max_bonds_group_2,
                         unsigned int bond_type);

      //! Destructor
      virtual ~DynamicBondUpdaterGPU();


    protected:
          //! filter out existing and doublicate bonds from all found possible bonds
          virtual void filterPossibleBonds();
          //! Update the vectors for traversal of pbc images
          virtual void updateImageVectors();
          //! Build the LBVH using the neighbor library
          virtual void buildTree();
          //! Traverse the LBVH using the neighbor library
          virtual void traverseTree();

    private:

        std::shared_ptr<Autotuner<1>> m_tuner_copy_nlist;     //!< Tuner for the primitive-copy kernel
        std::shared_ptr<Autotuner<1>> m_tuner_filter_bonds;   //!< Tuner for existing bond filter

        GPUFlags<int> m_num_nonzero_bonds_flag;            //!< GPU flag for the number of valid bonds
        GPUFlags<unsigned int> m_max_bonds_overflow_flag;  //!< GPU flag for overflow

        std::vector<std::unique_ptr<kernel::LBVHWrapper>> m_lbvh; //!< Array of LBVHs per-type
        std::vector<std::unique_ptr<kernel::LBVHTraverserWrapper>>  m_traverser;     //!< Array of LBVH traverers per-type
        GPUVector<Scalar3> m_image_list;               //!< List of translation vectors for traversal
        unsigned int m_n_images;                          //!< Number of translation vectors for traversal


        //! Compute the LBVH domain from the current box
        BoxDim getLBVHBox() const
            {
            const BoxDim& box = m_pdata->getBox();

            // ghost layer padding
            Scalar ghost_layer_width(0.0);
            #ifdef ENABLE_MPI
            if (m_comm) ghost_layer_width = m_comm->getGhostLayerMaxWidth();
            #endif

            Scalar3 ghost_width = make_scalar3(0.0, 0.0, 0.0);
            if (!box.getPeriodic().x) ghost_width.x = ghost_layer_width;
            if (!box.getPeriodic().y) ghost_width.y = ghost_layer_width;
            if (!box.getPeriodic().z && m_sysdef->getNDimensions() == 3) ghost_width.z = ghost_layer_width;

            return BoxDim(box.getLo()-ghost_width, box.getHi()+ghost_width, box.getPeriodic());
            }


    };

namespace detail
{
//! Export DynamicBondUpdaterGPU to python
void export_DynamicBondUpdaterGPU(pybind11::module& m);
} // end namespace detail

} // end namespace azplugins
} // end namespace hoomd

#endif // AZPLUGINS_DYNAMIC_BOND_UPDATER_GPU_H_
