// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file DynamicBondUpdater.h
 * \brief Declaration of DynamicBondUpdater
 */

#ifndef AZPLUGINS_DYNAMIC_BOND_UPDATER_H_
#define AZPLUGINS_DYNAMIC_BOND_UPDATER_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/md/NeighborList.h"
#include "hoomd/Updater.h"
#include "hoomd/ParticleGroup.h"

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace azplugins
{

//! Particle type updater
/*!
 * Flips particle types based on their z height. Particles are classified as
 * either inside or outside of the region, and can be flipped between these two
 * types. Particles that are of neither the inside nor outside type are ignored.
 *
 * The region is defined by a slab along z. This could be easily extended to
 * accommodate a generic region criteria, but for now, the planar slab in z is
 * all that is necessary.
 */
class PYBIND11_EXPORT DynamicBondUpdater : public Updater
    {
    public:

        //! Constructor with parameters
        DynamicBondUpdater(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<NeighborList> nlist,
                    std::shared_ptr<ParticleGroup> group_1,
                    std::shared_ptr<ParticleGroup> group_2,
                    const Scalar r_cutsq,
                    unsigned int bond_type,
                    unsigned int bond_reservoir_type,
                    unsigned int max_bonds_group_1,
                    unsigned int max_bonds_group_2);

        //! Destructor
        virtual ~DynamicBondUpdater();

        //! Evaporate particles
        virtual void update(unsigned int timestep);


    protected:
        std::shared_ptr<NeighborList> m_nlist;    //!< The neighborlist to use for bond finding
        std::shared_ptr<BondData> m_bond_data;    //!< Bond data

        std::shared_ptr<ParticleGroup> m_group_1;   //!< First particle group to form bonds with
        std::shared_ptr<ParticleGroup> m_group_2;   //!< Second particle group to form bonds with

        const Scalar m_r_cutsq; //!< cutoff squared for the bond forming criterion

        unsigned int m_bond_type;              //!< Type id of the bond to form
        unsigned int m_bond_reservoir_type;    //!< Type id of the bond reservoir

        unsigned int m_max_bonds_group_1; //!< maximum number of bonds which can be formed by the first group
        unsigned int m_max_bonds_group_2; //!< maximum number of bonds which can be formed by the second group

        GPUArray<unsigned int> m_curr_num_bonds; //!< current number of bonds for each particle
        std::map<std::pair<int, int>, int> m_all_existing_bonds;     //!< map of all current existing bonds of bond_type
        GPUArray<Scalar2> m_possible_bonds;   //!< list of possible bonds, size:  size(group_1)*max_bonds_1

        unsigned int m_curr_bonds_to_form; //!< number of bonds to form in the current timestep
        unsigned int m_reservoir_size;

        //! Changes the particle types according to an update rule
        virtual void findPotentialBondPairs(unsigned int timestep);

        virtual void formBondPairs(unsigned int timestep);

    private:
        void calculateCurrentBonds();
        void checkSystemSetup();
    };

namespace detail
{
//! Export the Evaporator to python
void export_DynamicBondUpdater(pybind11::module& m);
} // end namespace detail

} // end namespace azplugins

#endif // AZPLUGINS_TYPE_UPDATER_H_
