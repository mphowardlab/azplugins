// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file ParticleEvaporator.h
 * \brief Declaration of ParticleEvaporator
 */

#ifndef AZPLUGINS_PARTICLE_EVAPORATOR_H_
#define AZPLUGINS_PARTICLE_EVAPORATOR_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "TypeUpdater.h"

namespace azplugins
    {

//! Solvent-particle evaporator
/*!
 * Evaporates solvent particles by flipping their particle type from
 * solvent to evaporated. When solvent particles are evaporated, they are
 * randomly migrated to a new location in the global simulation box.
 * This migration is necessary in order to prevent a build-up of particles in one
 * region of the box, which will cause a crash of the CellList. The evaporated
 * particles should not participate in any energetic interactions with other
 * particles. This is more efficient than actually deleting the particles, which
 * is not a supported HOOMD operation.
 *
 * \warning This updater should not be used in conjunction with a Nose-Hoover
 *          style thermostat or barostat, which work under assumption of constant
 *          number of coupled particles.
 *
 * \warning The temperature reported by ComputeThermo will likely not be accurate
 *          during evaporation for two reasons: (1) the system is out-of-equilibrium,
 *          and so may experience a net convective drift, and (2) the evaporated
 *          particles are retained in the system as degrees-of-freedom, but they
 *          are typically uncoupled (non-interacting). Hence, their velocities
 *          make a meaningless contribution to the reported temperature.
 */
class PYBIND11_EXPORT ParticleEvaporator : public TypeUpdater
    {
    public:
    //! Simple constructor
    ParticleEvaporator(std::shared_ptr<SystemDefinition> sysdef, unsigned int seed);

    //! Constructor with parameters
    ParticleEvaporator(std::shared_ptr<SystemDefinition> sysdef,
                       unsigned int evap_type,
                       unsigned int solvent_type,
                       Scalar z_lo,
                       Scalar z_hi,
                       unsigned int seed);

    //! Destructor
    virtual ~ParticleEvaporator() {};

    //! Get the maximum number of particles to evaporate
    unsigned int getNEvapMax() const
        {
        return m_Nevap_max;
        }

    //! Set the maximum number of particles to evaporate
    void setNEvapMax(unsigned int Nevap_max)
        {
        m_Nevap_max = Nevap_max;
        }

    protected:
    unsigned int m_seed;             //!< Seed to evaporator pseudo-random number generator
    unsigned int m_Nevap_max;        //!< Maximum number of particles to evaporate
    unsigned int m_Npick;            //!< Number of particles picked for evaporation on this rank
    GPUVector<unsigned int> m_picks; //!< Particles picked for evaporation on this rank
    GPUVector<unsigned int> m_mark;  //!< Indexes of atoms that can be deleted

    //! Changes the particle types according to an update rule on the GPU
    virtual void changeTypes(unsigned int timestep);

    //! Mark particles as candidates for evaporation
    virtual unsigned int markParticles();

    //! Apply evaporation to picks
    virtual void applyPicks();

    private:
    std::vector<unsigned int> m_all_picks; //!< All picked particles

    //! Make a random pick of particles across all ranks
    void makeAllPicks(unsigned int timestep, unsigned int N_pick, unsigned N_mark_total);
    };

namespace detail
    {
//! Export ParticleEvaporator to python
void export_ParticleEvaporator(pybind11::module& m);
    } // end namespace detail

    } // end namespace azplugins

#endif // AZPLUGINS_PARTICLE_EVAPORATOR_H_
