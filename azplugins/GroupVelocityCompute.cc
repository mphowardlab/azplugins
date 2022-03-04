// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file GroupVelocityCompute.cc
 * \brief Definition of GroupVelocityCompute
 */

#include "GroupVelocityCompute.h"

namespace azplugins
{

/*!
 * \param sysdef System definition
 * \param group Particle group
 * \param suffix Suffix to attach to logged quantities
 */
GroupVelocityCompute::GroupVelocityCompute(std::shared_ptr<SystemDefinition> sysdef,
                                           std::shared_ptr<ParticleGroup> group,
                                           const std::string& suffix)
    : Compute(sysdef),
      m_group(group),
      m_lognames{"vx"+suffix,"vy"+suffix,"vz"+suffix},
      m_velocity(0,0,0)
    {
    }

/*!
 * \param timestep Simulation timestep
 *
 * The center-of-mass velocity of the group is determined by first summing the
 * momentum and mass of all particles, then dividing momentum by mass. This
 * compute supports MPI decomposition, and the result is available on all ranks.
 */
void GroupVelocityCompute::compute(unsigned int timestep)
    {
    if (!shouldCompute(timestep))
        return;

    // empty group has no velocity, quit early
    if (m_group->getNumMembersGlobal() == 0)
        {
        m_velocity = make_scalar3(0,0,0);
        return;
        }

    // accumulate momentum and masses
    const unsigned int N = m_group->getNumMembers();
    ArrayHandle<unsigned int> h_index(m_group->getIndexArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    Scalar4 mom_mass(0,0,0,0);
    for (unsigned int i=0; i < N; ++i)
        {
        const Scalar4 vel_mass = h_vel.data[h_index.data[i]];
        const Scalar mass = vel_mass.w;
        mom_mass += make_scalar4(mass*vel_mass.x,mass*vel_mass.y,mass*vel_mass.z,mass);
        }
    #ifdef ENABLE_MPI
    if (m_comm)
        {
        Scalar buffer[4] = {mom_mass.x, mom_mass.y, mom_mass.z, mom_mass.w};
        MPI_Allreduce(MPI_IN_PLACE, &buffer[0], 4, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        mom_mass = make_scalar4(buffer[0],buffer[1],buffer[2],buffer[3]);
        }
    #endif // ENABLE_MPI

    // reduce total momentum by mass to get center-of-mass
    m_velocity = make_scalar3(mom_mass.x,mom_mass.y,mom_mass.z)/mom_mass.w;
    }

std::vector<std::string> GroupVelocityCompute::getProvidedLogQuantities()
	{
	return m_lognames;
	}

Scalar GroupVelocityCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    compute(timestep);
    if (quantity == m_lognames[0])
        {
        return m_velocity.x;
        }
    else if (quantity == m_lognames[1])
        {
        return m_velocity.y;
        }
    else if (quantity == m_lognames[2])
        {
        return m_velocity.z;
        }
    else
        {
        m_exec_conf->msg->error() << "GroupVelocityCompute: " << quantity << " is not a valid log quantity" << endl;
        throw runtime_error("Unknown log quantity");
        }
    }
    
namespace detail
{
void export_GroupVelocityCompute(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<GroupVelocityCompute,std::shared_ptr<GroupVelocityCompute>,Compute>(m, "GroupVelocityCompute")
        .def(py::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<ParticleGroup>,const std::string&>())
        ;
    }
} // end namespace detail
} // end namespace azplugins