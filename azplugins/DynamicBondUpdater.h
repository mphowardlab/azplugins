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

#include "hoomd/AABBTree.h"
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
                    std::shared_ptr<ParticleGroup> group_1,
                    std::shared_ptr<ParticleGroup> group_2,
                    const Scalar r_cut,
                    unsigned int bond_type,
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

        const Scalar m_r_cut; //!< cutoff for the bond forming criterion

        unsigned int m_bond_type;              //!< Type id of the bond to form

        unsigned int m_max_bonds_group_1; //!< maximum number of bonds which can be formed by the first group
        unsigned int m_max_bonds_group_2; //!< maximum number of bonds which can be formed by the second group

        unsigned int m_max_bonds; //!< maximum number of possible bonds found
        unsigned int m_max_bonds_overflow; //!< maximum number of possible bonds found if there is an overflow


        GPUArray<Scalar3> m_all_possible_bonds;   //!< list of possible bonds, size:  size(group_1)*n_max_bonds(=20)
        unsigned int m_all_possible_bonds_end;

        unsigned int m_curr_bonds_to_form; //!< number of bonds to form in the current timestep
        unsigned int m_reservoir_size;

        hpmc::detail::AABBTree        m_aabb_tree;     //!<  AABB tree for group 1
        GPUVector<hpmc::detail::AABB> m_aabbs;          //!< Flat array of AABBs of all types

        std::vector< vec3<Scalar> > m_image_list;    //!< List of translation vectors
        unsigned int m_n_images;                //!< The number of image vectors to check


        GPUArray<unsigned int> m_existing_bonds_list;  //!< List of existing bonded particles referenced by tag
        GPUArray<unsigned int> m_n_existing_bonds;    //!< Number of existing bonds for a given particle tag
        unsigned int m_max_existing_bonds_list; //!< maximum number of  bonds in list of existing bonded particles
        Index2D m_existing_bonds_list_indexer;         //!< Indexer for accessing the by-tag bonded particle list

    private:
        //bool SortBonds(Scalar3 i, Scalar3 j); //todo: should go into helper class. or not be member of this class anyway
        //bool CompareBonds(Scalar3 i, Scalar3 j); //todo: should go into helper class. or not be member of this class anyway
        bool CheckisExistingLegalBond(Scalar3 i); //this acesses  info in m_existing_bonds_list_tag todo: rename to something sensible
        void calculateExistingBonds();
        void calculatePossibleBonds();
        void filterPossibleBonds();
        void makeBonds();
        void AddtoExistingBonds(unsigned int tag1,unsigned int tag2);
        bool isExistingBond(unsigned int tag1,unsigned int tag2); //this acesses  info in m_existing_bonds_list_tag
        void updateImageVectors();
        void checkSystemSetup();
        void resizePossibleBondlists();
        void resizeExistingBondList();

        //! Notification of a box size change
        void slotBoxChanged()
            {
            m_box_changed = true;
            }

        bool m_box_changed;         //!< Flag if box changed

    };

namespace detail
{
//! Export the Evaporator to python
void export_DynamicBondUpdater(pybind11::module& m);
} // end namespace detail

} // end namespace azplugins

#endif // AZPLUGINS_TYPE_UPDATER_H_
