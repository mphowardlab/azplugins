// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file GroupVelocityCompute.h
 * \brief Declaration of GroupVelocityCompute
 */

#ifndef AZPLUGINS_GROUP_VELOCITY_COMPUTE_H_
#define AZPLUGINS_GROUP_VELOCITY_COMPUTE_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/Compute.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/SystemDefinition.h"
#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif // ENABLE_MPI

#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace hoomd
{
namespace azplugins
{
//! Compute the center-of-mass velocity of a group of particles
class PYBIND11_EXPORT GroupVelocityCompute : public Compute
    {
    public:
        //! Constructor
        GroupVelocityCompute(std::shared_ptr<hoomd::SystemDefinition> sysdef,
                             std::shared_ptr<hoomd::ParticleGroup> group,
                             const std::string& suffix);

        //! Destructor
        virtual ~GroupVelocityCompute();

        //! Compute center-of-mass velocity of group
        void compute(uint64_t timestep) override;

        //! List of logged quantities
        std::vector<std::string> getProvidedLogQuantities();

        //! Return the logged value
        Scalar getLogValue(const std::string& quantity, uint64_t timestep);

    protected:
        std::shared_ptr<ParticleGroup> m_group; //!< Particle group
        std::vector<std::string> m_lognames;    //!< Logged quantities
        Scalar3 m_velocity;                     //!< Last compute velocity
#ifdef ENABLE_MPI
        std::shared_ptr<Communicator> m_comm;   //!< Communicator
#endif // ENABLE_MPI
    };

namespace detail
{
//! Exports the GroupVelocityCompute to python
void export_GroupVelocityCompute(pybind11::module& m);
} // end namespace detail
} // end namespace azplugins
} // end namespace hoomd

#endif // AZPLUGINS_GROUP_VELOCITY_COMPUTE_H_
