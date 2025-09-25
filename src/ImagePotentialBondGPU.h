// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

// File modified from HOOMD-blue
// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_IMAGE_POTENTIAL_BOND_GPU_H_
#define AZPLUGINS_IMAGE_POTENTIAL_BOND_GPU_H_

#ifdef ENABLE_HIP

#include "ImagePotentialBondGPU.cuh"
#include "hoomd/Autotuner.h"
#include "hoomd/md/PotentialBondGPU.h"

/*! \file ImagePotentialBondGPU.h
    \brief Defines the template class for standard bond potentials on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {
namespace azplugins
    {
//! Template class for computing bond potentials on the GPU
/*!
    \tparam evaluator EvaluatorBond class used to evaluate V(r) and F(r)/r
    \tparam Bonds Bond data type

    \sa export_ImagePotentialBondGPU()
*/
template<class evaluator, class Bonds>
class ImagePotentialBondGPU : public md::PotentialBondGPU<evaluator, Bonds>
    {
    public:
    //! Inherit constructors from base class
    using md::PotentialBondGPU<evaluator, Bonds>::PotentialBondGPU;

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    CommFlags getRequestedCommFlags(uint64_t timestep) override;
#endif

    protected:
    void computeForces(uint64_t timestep) override;
    GPUArray<unsigned int> m_flags; //!< Flags set during the kernel execution
    };

template<class evaluator, class Bonds>
void ImagePotentialBondGPU<evaluator, Bonds>::computeForces(uint64_t timestep)
    {
    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar> d_charge(this->m_pdata->getCharges(),
                                 access_location::device,
                                 access_mode::read);
    ArrayHandle<int3> d_images(this->m_pdata->getImages(),
                               access_location::device,
                               access_mode::read);

    BoxDim box = this->m_pdata->getGlobalBox();

    // access parameters
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params,
                                                         access_location::device,
                                                         access_mode::read);

    // access net force & virial
    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::readwrite);

        {
        const GPUArray<typename Bonds::members_t>& gpu_bond_list = this->m_bond_data->getGPUTable();
        const Index2D& gpu_table_indexer = this->m_bond_data->getGPUTableIndexer();

        ArrayHandle<typename Bonds::members_t> d_gpu_bondlist(gpu_bond_list,
                                                              access_location::device,
                                                              access_mode::read);

        ArrayHandle<unsigned int> d_gpu_bond_pos_list(this->m_bond_data->getGPUPosTable(),
                                                      access_location::device,
                                                      access_mode::read);
        ArrayHandle<unsigned int> d_gpu_n_bonds(this->m_bond_data->getNGroupsArray(),
                                                access_location::device,
                                                access_mode::read);

        // access the flags array for overwriting
        ArrayHandle<unsigned int> d_flags(m_flags, access_location::device, access_mode::readwrite);

        this->m_tuner->begin();

        // Use Custom Kernel
        azplugins::gpu::compute_bond_forces<evaluator, 2>(
            azplugins::gpu::bond_args_t<Bonds::size>(d_force.data,
                                                     d_virial.data,
                                                     this->m_virial.getPitch(),
                                                     this->m_pdata->getN(),
                                                     this->m_pdata->getMaxN(),
                                                     d_pos.data,
                                                     d_charge.data,
                                                     d_images.data,
                                                     box,
                                                     d_gpu_bondlist.data,
                                                     gpu_table_indexer,
                                                     d_gpu_bond_pos_list.data,
                                                     d_gpu_n_bonds.data,
                                                     this->m_bond_data->getNTypes(),
                                                     this->m_tuner->getParam()[0],
                                                     this->m_exec_conf->dev_prop),
            d_params.data,
            d_flags.data);
        }

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        m_flags = GPUArray<unsigned int>(1, this->m_exec_conf);
        ArrayHandle<unsigned int> h_flags(m_flags, access_location::host, access_mode::read);

        if (h_flags.data[0] & 1)
            {
            this->m_exec_conf->msg->error()
                << "bond." << evaluator::getName() << ": bond out of bounds (" << h_flags.data[0]
                << ")" << std::endl;
            throw std::runtime_error("Error in bond calculation");
            }
        }
    this->m_tuner->end();
    }

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
template<class evaluator, class Bonds>
CommFlags ImagePotentialBondGPU<evaluator, Bonds>::getRequestedCommFlags(uint64_t timestep)
    {
    CommFlags flags = md::PotentialBondGPU<evaluator, Bonds>::getRequestedCommFlags(timestep);
    flags[comm_flag::image] = 1;

    return flags;
    }
#endif

namespace detail
    {
//! Exports the ImagePotentialBondGPU class to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class T> void export_ImagePotentialBondGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ImagePotentialBondGPU<T, BondData>,
                     md::PotentialBondGPU<T, BondData>,
                     std::shared_ptr<ImagePotentialBondGPU<T, BondData>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#endif // ENABLE_HIP
#endif // AZPLUGINS_IMAGE_POTENTIAL_BOND_GPU_H_
