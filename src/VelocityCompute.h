// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_VELOCITY_COMPUTE_H_
#define AZPLUGINS_VELOCITY_COMPUTE_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/Compute.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/SystemDefinition.h"

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace azplugins
    {
//! Compute the center-of-mass velocity of a group of particles
class PYBIND11_EXPORT VelocityCompute : public Compute
    {
    public:
    //! Constructor
    VelocityCompute(std::shared_ptr<hoomd::SystemDefinition> sysdef,
                    std::shared_ptr<hoomd::ParticleGroup> group,
                    bool include_mpcd_particles);

    //! Destructor
    virtual ~VelocityCompute();

    //! Compute center-of-mass velocity of group
    void compute(uint64_t timestep) override;

    //! Get particle group
    std::shared_ptr<ParticleGroup> getGroup() const
        {
        return m_group;
        }

    bool includeMPCDParticles() const
        {
        return m_include_mpcd_particles;
        }

    //! Get most recently computed velocity
    Scalar3 getVelocity() const
        {
        return m_velocity;
        }

    protected:
    std::shared_ptr<ParticleGroup> m_group; //!< Particle group
    bool m_include_mpcd_particles;          //!< Whether to include MPCD particles
    Scalar3 m_velocity;                     //!< Last computed velocity

    virtual void sumMomentumAndMass(Scalar3& momentum, Scalar& mass);
    };

    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_VELOCITY_COMPUTE_H_
