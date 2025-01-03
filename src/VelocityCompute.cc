// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "VelocityCompute.h"
#include "ParticleDataLoader.h"

namespace hoomd
    {
namespace azplugins
    {

/*!
 * \param sysdef System definition
 * \param group Particle group
 */
VelocityCompute::VelocityCompute(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<ParticleGroup> group,
                                 bool include_mpcd_particles)
    : Compute(sysdef), m_group(group), m_include_mpcd_particles(include_mpcd_particles),
      m_velocity(make_scalar3(0, 0, 0))
    {
    }

VelocityCompute::~VelocityCompute() { }

template<class LoadOpT>
void accumulateParticle(Scalar& mass, Scalar3& momentum, const LoadOpT& load_op, unsigned int idx)
    {
    Scalar3 v;
    Scalar m;
    load_op(v, m, idx);

    mass += m;
    momentum += m * v;
    }

/*!
 * \param timestep Simulation timestep
 *
 * The center-of-mass velocity of the group is determined by first summing the
 * momentum and mass of all particles, then dividing momentum by mass. This
 * compute supports MPI decomposition, and the result is available on all ranks.
 */
void VelocityCompute::compute(uint64_t timestep)
    {
    if (!shouldCompute(timestep))
        return;

    Scalar3 momentum = make_scalar3(0, 0, 0);
    Scalar mass(0);

    if (m_group)
        {
        const unsigned int N = m_group->getNumMembers();
        ArrayHandle<unsigned int> h_index(m_group->getIndexArray(),
                                          access_location::host,
                                          access_mode::read);
        ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        detail::LoadHOOMDGroupVelocityMass load_op(h_vel.data, h_index.data);

        for (unsigned int i = 0; i < N; ++i)
            {
            accumulateParticle(mass, momentum, load_op, i);
            }
        }

#ifdef BUILD_MPCD
    if (m_include_mpcd_particles)
        {
        auto mpcd_pdata = m_sysdef->getMPCDParticleData();
        const unsigned int N = mpcd_pdata->getN();
        ArrayHandle<Scalar4> h_vel(mpcd_pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        detail::LoadMPCDVelocityMass load_op(h_vel.data, mpcd_pdata->getMass());

        for (unsigned int i = 0; i < N; ++i)
            {
            accumulateParticle(mass, momentum, load_op, i);
            }
        }
#endif // BUILD_MPCD

#ifdef ENABLE_MPI
    if (m_exec_conf->getNRanks() > 1)
        {
        Scalar buffer[4] = {momentum.x, momentum.y, momentum.z, mass};
        MPI_Allreduce(MPI_IN_PLACE,
                      &buffer[0],
                      4,
                      MPI_HOOMD_SCALAR,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        momentum = make_scalar3(buffer[0], buffer[1], buffer[2]);
        mass = buffer[3];
        }
#endif // ENABLE_MPI

    // reduce total momentum by mass to get center-of-mass velocity
    if (mass > 0)
        {
        m_velocity = momentum / mass;
        }
    else
        {
        m_velocity = make_scalar3(0, 0, 0);
        }
    }

namespace detail
    {
void export_VelocityCompute(pybind11::module& m)
    {
    pybind11::class_<VelocityCompute, Compute, std::shared_ptr<VelocityCompute>>(m,
                                                                                 "VelocityCompute")
        .def(pybind11::
                 init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, bool>())
        .def_property_readonly("filter",
                               [](const VelocityCompute& self)
                               {
                                   auto group = self.getGroup();
                                   return (group) ? group->getFilter()
                                                  : std::shared_ptr<hoomd::ParticleFilter>();
                               })
        .def_property_readonly("include_mpcd_particles", &VelocityCompute::includeMPCDParticles)
        .def_property_readonly("velocity",
                               [](const VelocityCompute& self)
                               {
                                   const auto v = self.getVelocity();
                                   return pybind11::make_tuple(v.x, v.y, v.z);
                               });
    }
    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd
