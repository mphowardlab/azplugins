// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file MPCDVelocityCompute.cc
 * \brief Definition of MPCDVelocityCompute
 */

#include "MPCDVelocityCompute.h"

namespace azplugins
    {

/*!
 * \param sysdata MPCD system data
 * \param suffix Suffix to attach to logged quantities
 */
MPCDVelocityCompute::MPCDVelocityCompute(std::shared_ptr<mpcd::SystemData> sysdata,
                                         const std::string& suffix)
    : Compute(sysdata->getSystemDefinition()), m_mpcd_pdata(sysdata->getParticleData()),
      m_lognames {"mpcd_vx" + suffix, "mpcd_vy" + suffix, "mpcd_vz" + suffix},
      m_velocity(make_scalar3(0, 0, 0))
    {
    }

MPCDVelocityCompute::~MPCDVelocityCompute() { }

/*!
 * \param timestep Simulation timestep
 *
 * The center-of-mass velocity of the MPCD particles is determined by first summing the
 * momentum and mass of all particles, then dividing momentum by mass. This
 * compute supports MPI decomposition, and the result is available on all ranks.
 */
void MPCDVelocityCompute::compute(unsigned int timestep)
    {
    if (!shouldCompute(timestep))
        return;

    // empty particle data has no velocity, quit early
    if (m_mpcd_pdata->getNGlobal() == 0)
        {
        m_velocity = make_scalar3(0, 0, 0);
        return;
        }

    // accumulate momentum and masses
    // this could be simplified by assuming mass is the same, but to make intention clear in
    // case particles can have different masses in future, carrying out the explicit calculation
    const unsigned int N = m_mpcd_pdata->getN();
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(),
                               access_location::host,
                               access_mode::read);
    const Scalar m = m_mpcd_pdata->getMass();
    Scalar3 momentum = make_scalar3(0, 0, 0);
    Scalar mass(0);
    for (unsigned int i = 0; i < N; ++i)
        {
        const Scalar4 vel_cell = h_vel.data[i];
        const Scalar3 vel = make_scalar3(vel_cell.x, vel_cell.y, vel_cell.z);
        momentum += m * vel;
        mass += m;
        }
#ifdef ENABLE_MPI
    if (m_comm)
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

    // reduce total momentum by mass to get center-of-mass
    m_velocity = momentum / mass;
    }

std::vector<std::string> MPCDVelocityCompute::getProvidedLogQuantities()
    {
    return m_lognames;
    }

Scalar MPCDVelocityCompute::getLogValue(const std::string& quantity, unsigned int timestep)
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
        m_exec_conf->msg->error() << "MPCDVelocityCompute: " << quantity
                                  << " is not a valid log quantity" << std::endl;
        throw std::runtime_error("Unknown log quantity");
        }
    }

namespace detail
    {
void export_MPCDVelocityCompute(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<MPCDVelocityCompute, std::shared_ptr<MPCDVelocityCompute>>(m,
                                                                          "MPCDVelocityCompute",
                                                                          py::base<Compute>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>, const std::string&>());
    }
    } // end namespace detail
    } // end namespace azplugins
