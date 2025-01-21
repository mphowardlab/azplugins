// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_PARTICLE_DATA_LOADER_H_
#define AZPLUGINS_PARTICLE_DATA_LOADER_H_

#include "hoomd/HOOMDMath.h"

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif // __HIPCC__

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {

//! Load HOOMD particle in a group from an index
class LoadHOOMDGroupPositionVelocityMass
    {
    public:
    HOSTDEVICE
    LoadHOOMDGroupPositionVelocityMass(const Scalar4* positions,
                                       const Scalar4* velocities,
                                       const unsigned int* indexes)
        : m_positions(positions), m_velocities(velocities), m_indexes(indexes)
        {
        }

    HOSTDEVICE void
    operator()(Scalar3& position, Scalar3& velocity, Scalar& mass, unsigned int idx) const
        {
        const unsigned int pidx = m_indexes[idx];
        const Scalar4 postype = m_positions[pidx];
        position = make_scalar3(postype.x, postype.y, postype.z);

        const Scalar4 velmass = m_velocities[pidx];
        velocity = make_scalar3(velmass.x, velmass.y, velmass.z);
        mass = velmass.w;
        }

    private:
    const Scalar4* const m_positions;
    const Scalar4* const m_velocities;
    const unsigned int* const m_indexes;
    };

//! Load MPCD particle from an index
class LoadMPCDPositionVelocityMass
    {
    public:
    HOSTDEVICE
    LoadMPCDPositionVelocityMass(const Scalar4* positions,
                                 const Scalar4* velocities,
                                 const Scalar mass)
        : m_positions(positions), m_velocities(velocities), m_mass(mass)
        {
        }

    HOSTDEVICE void
    operator()(Scalar3& position, Scalar3& velocity, Scalar& mass, unsigned int idx) const
        {
        const Scalar4 postype = m_positions[idx];
        position = make_scalar3(postype.x, postype.y, postype.z);

        const Scalar4 velcell = m_velocities[idx];
        velocity = make_scalar3(velcell.x, velcell.y, velcell.z);
        mass = m_mass;
        }

    private:
    const Scalar4* const m_positions;
    const Scalar4* const m_velocities;
    const Scalar m_mass;
    };

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#undef HOSTDEVICE

#endif // AZPLUGINS_PARTICLE_DATA_LOADER_H_
