// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

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

class PYBIND11_EXPORT DynamicBondUpdater : public Updater
    {
    public:

        //! Constructor with parameters
        DynamicBondUpdater(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<NeighborList> nlist,
                           std::shared_ptr<ParticleGroup> group_1,
                           std::shared_ptr<ParticleGroup> group_2,
                           const Scalar r_cut,
                           unsigned int bond_type,
                           unsigned int max_bonds_group_1,
                           unsigned int max_bonds_group_2);

        //! Destructor
        virtual ~DynamicBondUpdater();

        //! update
        virtual void update(unsigned int timestep);


    protected:
        std::shared_ptr<NeighborList> m_nlist;    //!< The neighborlist to use for bond finding
        std::shared_ptr<BondData> m_bond_data;    //!< Bond data

        std::shared_ptr<ParticleGroup> m_group_1;   //!< First particle group to form bonds with
        std::shared_ptr<ParticleGroup> m_group_2;   //!< Second particle group to form bonds with

        const Scalar m_r_cut;                       //!< cutoff for the bond forming criterion
        const unsigned int m_bond_type;                   //!< Type id of the bond to form
        const unsigned int m_max_bonds_group_1;           //!< maximum number of bonds which can be formed by the first group
        const unsigned int m_max_bonds_group_2;           //!< maximum number of bonds which can be formed by the second group

        unsigned int m_max_bonds;                   //!<  maximum number of possible bonds which can be found
        unsigned int m_max_bonds_overflow;          //!< registers if there is an overflow in  maximum number of possible bonds

        GPUArray<Scalar3> m_all_possible_bonds;   //!< list of possible bonds, size:  size(group_1)*m_max_bonds
        unsigned int m_num_all_possible_bonds;    //!< number of valid possible bonds at the beginning of m_all_possible_bonds

        hpmc::detail::AABBTree        m_aabb_tree;  //!< AABB tree for group_2
        GPUVector<hpmc::detail::AABB> m_aabbs;      //!< Flat array of AABBs of all types
        std::vector< vec3<Scalar> > m_image_list;   //!< List of translation vectors for tree traversal
        unsigned int m_n_images;                    //!< The number of image vectors to check

        GPUArray<unsigned int> m_existing_bonds_list;  //!< List of existing bonded particles referenced by tag
        GPUArray<unsigned int> m_n_existing_bonds;     //!< Number of existing bonds for a given particle tag
        unsigned int m_max_existing_bonds_list;        //!< maximum number of  bonds in list of existing bonded particles
        Index2D m_existing_bonds_list_indexer;         //!< Indexer for accessing the by-tag bonded particle list
        virtual void filterPossibleBonds();

        bool CheckisExistingLegalBond(Scalar3 i); //this acesses info in m_existing_bonds_list_tag. todo: rename to something sensible
        void calculateExistingBonds();

        virtual void buildTree();
        virtual void traverseTree();

        void makeBonds();
        void AddtoExistingBonds(unsigned int tag1,unsigned int tag2);
        bool isExistingBond(unsigned int tag1,unsigned int tag2); //this acesses info in m_existing_bonds_list_tag
        virtual void updateImageVectors();
        void checkSystemSetup();
        virtual void resizePossibleBondlists();
        void resizeExistingBondList();
        virtual void allocateParticleArrays();

        //! Notification of a box size change
        void slotBoxChanged()
            {
            m_box_changed = true;
            }

        //! Notification of total particle number change
        void slotNumParticlesChanged()
            {
            m_max_N_changed = true;
            }

        bool m_box_changed;          //!< Flag if box dimensions changed
        bool m_max_N_changed;        //!< Flag if total number of particles changed

    };

namespace detail
{
//! Export the Evaporator to python
void export_DynamicBondUpdater(pybind11::module& m);
} // end namespace detail

} // end namespace azplugins

#endif // AZPLUGINS_TYPE_UPDATER_H_
