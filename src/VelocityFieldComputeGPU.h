// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_VELOCITY_FIELD_COMPUTE_GPU_H_
#define AZPLUGINS_VELOCITY_FIELD_COMPUTE_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "VelocityFieldCompute.h"
#include "VelocityFieldComputeGPU.cuh"
#include "hoomd/Autotuner.h"

namespace hoomd
    {
namespace azplugins
    {
//! Compute a velocity field in a region of space using histograms
template<class BinOpT>
class PYBIND11_EXPORT VelocityFieldComputeGPU : public VelocityFieldCompute<BinOpT>
    {
    public:
    //! Constructor
    VelocityFieldComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                            uint3 num_bins,
                            Scalar3 lower_bounds,
                            Scalar3 upper_bounds,
                            std::shared_ptr<ParticleGroup> group,
                            bool include_mpcd_particles)
        : VelocityFieldCompute<BinOpT>(sysdef,
                                       num_bins,
                                       lower_bounds,
                                       upper_bounds,
                                       group,
                                       include_mpcd_particles)
        {
        if (this->m_group)
            {
            m_tuner_hoomd.reset(
                new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                 this->m_exec_conf,
                                 "velocity_field_hoomd"));
            this->m_autotuners.push_back(m_tuner_hoomd);
            }

#ifdef BUILD_MPCD
        if (this->m_include_mpcd_particles)
            {
            m_tuner_mpcd.reset(
                new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                 this->m_exec_conf,
                                 "velocity_field_mpcd"));
            this->m_autotuners.push_back(m_tuner_mpcd);
            }
#endif // BUILD_MPCD
        }

    protected:
    void binParticles() override;

    private:
    std::shared_ptr<Autotuner<1>> m_tuner_hoomd; //!< Tuner for HOOMD particles
#ifdef BUILD_MPCD
    std::shared_ptr<Autotuner<1>> m_tuner_mpcd; //!< Tuner for MPCD particles
#endif
    };

template<class BinOpT> void VelocityFieldComputeGPU<BinOpT>::binParticles()
    {
    const BinOpT& bin_op = *this->m_binning_op;
    ArrayHandle<Scalar> d_mass(this->m_mass, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar3> d_momentum(this->m_momentum,
                                    access_location::device,
                                    access_mode::overwrite);

    const size_t total_num_bins = bin_op.getTotalNumBins();
    gpu::zeroVelocityFieldArrays(d_mass.data, d_momentum.data, total_num_bins);
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (this->m_group)
        {
        ArrayHandle<unsigned int> d_index(this->m_group->getIndexArray(),
                                          access_location::device,
                                          access_mode::read);
        ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
        ArrayHandle<Scalar4> d_vel(this->m_pdata->getVelocities(),
                                   access_location::device,
                                   access_mode::read);
        detail::LoadParticleGroupPositionVelocityMass load_op(d_pos.data, d_vel.data, d_index.data);
        const BoxDim& global_box = this->m_pdata->getGlobalBox();

        m_tuner_hoomd->begin();
        gpu::bin_velocity_field(d_mass.data,
                                d_momentum.data,
                                load_op,
                                bin_op,
                                global_box,
                                this->m_group->getNumMembers(),
                                m_tuner_hoomd->getParam()[0]);
        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_hoomd->end();
        }

#ifdef BUILD_MPCD
    if (this->m_include_mpcd_particles)
        {
        auto mpcd_pdata = this->m_sysdef->getMPCDParticleData();
        ArrayHandle<Scalar4> d_pos(mpcd_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
        ArrayHandle<Scalar4> d_vel(mpcd_pdata->getVelocities(),
                                   access_location::device,
                                   access_mode::read);
        detail::LoadMPCDParticlePositionVelocityMass load_op(d_pos.data,
                                                             d_vel.data,
                                                             mpcd_pdata->getMass());
        const BoxDim& global_box = this->m_pdata->getGlobalBox();

        m_tuner_mpcd->begin();
        gpu::bin_velocity_field(d_mass.data,
                                d_momentum.data,
                                load_op,
                                bin_op,
                                global_box,
                                mpcd_pdata->getN(),
                                m_tuner_mpcd->getParam()[0]);
        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_mpcd->end();
        }
#endif // BUILD_MPCD
    }

namespace detail
    {
template<class BinOpT>
void export_VelocityFieldComputeGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<VelocityFieldComputeGPU<BinOpT>,
                     VelocityFieldCompute<BinOpT>,
                     std::shared_ptr<VelocityFieldComputeGPU<BinOpT>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            uint3,
                            Scalar3,
                            Scalar3,
                            std::shared_ptr<ParticleGroup>,
                            bool>());
    }
    } // end namespace detail

    } // end namespace azplugins
    } // end namespace hoomd
#endif // AZPLUGINS_VELOCITY_FIELD_COMPUTE_GPU_H_
