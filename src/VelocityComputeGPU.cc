// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "VelocityComputeGPU.h"
#include "VelocityComputeGPU.cuh"

namespace hoomd
    {
namespace azplugins
    {

void VelocityComputeGPU::sumMomentumAndMass(Scalar3& momentum, Scalar& mass)
    {
    momentum = make_scalar3(0, 0, 0);
    mass = Scalar(0);

    if (m_group)
        {
        ArrayHandle<unsigned int> d_index(m_group->getIndexArray(),
                                          access_location::device,
                                          access_mode::read);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                                   access_location::device,
                                   access_mode::read);
        detail::LoadParticleGroupVelocityMass load_op(d_vel.data, d_index.data);

        Scalar3 group_momentum = make_scalar3(0, 0, 0);
        Scalar group_mass(0);
        gpu::sum_momentum_and_mass(group_momentum, group_mass, load_op, m_group->getNumMembers());

        momentum += group_momentum;
        mass += group_mass;
        }

#ifdef BUILD_MPCD
    if (m_include_mpcd_particles)
        {
        auto mpcd_pdata = m_sysdef->getMPCDParticleData();
        ArrayHandle<Scalar4> d_vel(mpcd_pdata->getVelocities(),
                                   access_location::device,
                                   access_mode::read);
        detail::LoadMPCDParticleVelocityMass load_op(d_vel.data, mpcd_pdata->getMass());

        Scalar3 mpcd_momentum = make_scalar3(0, 0, 0);
        Scalar mpcd_mass(0);
        gpu::sum_momentum_and_mass(mpcd_momentum, mpcd_mass, load_op, mpcd_pdata->getN());

        momentum += mpcd_momentum;
        mass += mpcd_mass;
        }
#endif // BUILD_MPCD
    }

namespace detail
    {
void export_VelocityComputeGPU(pybind11::module& m)
    {
    pybind11::class_<VelocityComputeGPU, VelocityCompute, std::shared_ptr<VelocityComputeGPU>>(
        m,
        "VelocityComputeGPU")
        .def(pybind11::
                 init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, bool>());
    }
    } // end namespace detail

    } // end namespace azplugins
    } // end namespace hoomd
