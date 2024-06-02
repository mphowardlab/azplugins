// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file MPCDVelocityCompute.h
 * \brief Declaration of MPCDVelocityCompute
 */

#ifndef AZPLUGINS_MPCD_VELOCITY_COMPUTE_H_
#define AZPLUGINS_MPCD_VELOCITY_COMPUTE_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/Compute.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/mpcd/ParticleData.h"
#include "hoomd/mpcd/SystemData.h"

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
#include <string>
#include <vector>

namespace azplugins
    {
//! Compute the center-of-mass velocity of MPCD particles
class PYBIND11_EXPORT MPCDVelocityCompute : public Compute
    {
    public:
    //! Constructor
    MPCDVelocityCompute(std::shared_ptr<mpcd::SystemData> sysdata, const std::string& suffix);

    //! Destructor
    ~MPCDVelocityCompute();

    //! Compute center-of-mass velocity of particles
    void compute(unsigned int timestep) override;

    //! List of logged quantities
    std::vector<std::string> getProvidedLogQuantities() override;

    //! Return the logged value
    Scalar getLogValue(const std::string& quantity, unsigned int timestep) override;

    protected:
    std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata; //!< Particle group
    std::vector<std::string> m_lognames;              //!< Logged quantities
    Scalar3 m_velocity;                               //!< Last compute velocity
    };

namespace detail
    {
//! Exports the MPCDVelocityCompute to python
void export_MPCDVelocityCompute(pybind11::module& m);
    } // end namespace detail
    } // end namespace azplugins

#endif // AZPLUGINS_MPCD_VELOCITY_COMPUTE_H_
