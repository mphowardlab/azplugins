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

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

#include "hoomd/AABBTree.h"
#include "hoomd/md/NeighborList.h"
#include "hoomd/Updater.h"
#include "hoomd/ParticleGroup.h"


#ifndef __HIPCC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif // __HIPCC__

#ifndef PYBIND11_EXPORT
#define PYBIND11_EXPORT __attribute__((visibility("default")))
#endif


namespace hoomd
{

namespace azplugins
{

class PYBIND11_EXPORT DynamicBondUpdater : public hoomd::Updater
    {
    public:

      //! Simple constructor
      DynamicBondUpdater(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<Trigger> trigger,
                         std::shared_ptr<ParticleGroup> group_1,
                         std::shared_ptr<ParticleGroup> group_2,
                         uint16_t seed);

      //! Constructor with parameters
      DynamicBondUpdater(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<Trigger> trigger,
                         std::shared_ptr<md::NeighborList> pair_nlist,
                         std::shared_ptr<ParticleGroup> group_1,
                         std::shared_ptr<ParticleGroup> group_2,
                         const Scalar r_cut,
                         const Scalar probability,
                         unsigned int bond_type,
                         unsigned int max_bonds_group_1,
                         unsigned int max_bonds_group_2,
                         uint16_t seed);

      //! Destructor
      virtual ~DynamicBondUpdater();

      //! update
      virtual void update(unsigned int timestep);

      //! Set the cutoff distance for finding bonds
      /*!
       * \param r_cut cutoff distance between particles for finding potential bonds
       */
      void setRcut(Scalar r_cut)
          {
          m_r_cut = r_cut;
          checkRcut();
          }
      //! Get the cutoff distance between particles for finding bonds
      Scalar getRcut()
          {
          return m_r_cut;
          }
      //! Set the bond type of the dynamically formed bonds
      /*!
       * \param bond_type type of bonds to be formed
       */
      void setBondType(unsigned int bond_type)
          {
          m_bond_type = bond_type;
          checkBondType();
          }
      //! Get the bond type of the dynamically formed bonds
      unsigned int getBondType()
          {
          return m_bond_type;
          }
      //! Set the maximum number of bonds on particles in group_1
      /*!
       * \param max_bonds_group_1 max number of bonds formed by particles in group 1
       */
      void setMaxBondsGroup1(unsigned int max_bonds_group_1)
          {
          m_max_bonds_group_1 = max_bonds_group_1;
          }
      //! Get the maximum number of bonds on particles in group_1
      unsigned int getMaxBondsGroup1()
          {
          return m_max_bonds_group_1;
          }
      //! Set the maximum number of bonds on particles in group_2
      /*!
       * \param max_bonds_group_2 max number of bonds formed by particles in group_2
       */
      void setMaxBondsGroup2(unsigned int max_bonds_group_2)
          {
          m_max_bonds_group_2 = max_bonds_group_2;
          }
      //! Get the maximum number of bonds on particles in group_2
      unsigned int getMaxBondsGroup2()
          {
          return m_max_bonds_group_2;
          }
      //! set probablilty
      void setProbability(Scalar probability)
          {
          m_probability = probability;
          }
      //! Get the probability
      Scalar getProbability()
          {
          return m_probability;
          }
      //! Set the hoomd neighbor list
      /*!
       * \param nlist hoomd NeighborList pointer
       */
      void setNeighbourList( std::shared_ptr<md::NeighborList> nlist)
          {
          m_pair_nlist = nlist;
          m_pair_nlist_exclusions_set = true;
          }

    protected:

        std::shared_ptr<BondData> m_bond_data;    //!< Bond data

        std::shared_ptr<ParticleGroup> m_group_1;   //!< First particle group to form bonds with
        std::shared_ptr<ParticleGroup> m_group_2;   //!< Second particle group to form bonds with
        bool m_groups_identical;

        Scalar m_r_cut;                             //!< cutoff for the bond forming criterion
        Scalar m_probability;                       //!< probability of bond formation if bond can be formed (i.e. within cutoff)
        unsigned int m_bond_type;                   //!< Type id of the bond to form
        unsigned int m_max_bonds_group_1;           //!< maximum number of bonds which can be formed by the first group
        unsigned int m_max_bonds_group_2;           //!< maximum number of bonds which can be formed by the second group
        uint16_t m_seed;                            //!< seed for random number generator for bond probability

        unsigned int m_max_bonds;                   //!<  maximum number of possible bonds (or neighbors) which can be found
        unsigned int m_max_bonds_overflow;          //!< registers if there is an overflow in  maximum number of possible bonds

        GPUArray<Scalar3> m_all_possible_bonds;     //!< list of possible bonds, size: NumMembers(group_1)*m_max_bonds
        unsigned int m_num_all_possible_bonds;      //!< number of valid possible bonds at the beginning of m_all_possible_bonds

        GPUArray<unsigned int> m_n_list;        //!< Neighbor list data
        GPUArray<unsigned int> m_n_neigh;       //!< Number of neighbors for each particle

        detail::AABBTree        m_aabb_tree;  //!< AABB tree for group_1
        GPUVector<detail::AABB> m_aabbs;      //!< Flat array of AABBs of particles in group_2
        std::vector< vec3<Scalar> > m_image_list;   //!< List of translation vectors for tree traversal
        unsigned int m_n_images;                    //!< The number of image vectors to check

        GPUArray<unsigned int> m_existing_bonds_list;  //!< List of existing bonded particles referenced by tag
        GPUArray<unsigned int> m_n_existing_bonds;     //!< Number of existing bonds for a given particle tag
        unsigned int m_max_existing_bonds_list;        //!< maximum number of  bonds in list of existing bonded particles
        Index2D m_existing_bonds_list_indexer;         //!< Indexer for accessing the by-tag bonded particle list

        std::shared_ptr<md::NeighborList> m_pair_nlist;    //!< The hoomd neighborlist, only used if exclusions of the newly formed bonds need to be set
        bool m_pair_nlist_exclusions_set;              //!< whether or not the bonds are set as exclusions in the hoomd particle neighborlist. Set to true when m_pair_nlist is set

        //! filter out existing and doublicate bonds from all found possible bonds
        virtual void filterPossibleBonds();
        //! build the neighbor list AABB tree
        virtual void buildTree();
        //! traverse the neighbor list ABB tree
        virtual void traverseTree();


        bool CheckisExistingLegalBond(Scalar3 i); //this acesses info in m_existing_bonds_list_tag. todo: rename to something sensible
        void calculateExistingBonds();
        void makeBonds(unsigned int timestep);

        void AddtoExistingBonds(unsigned int tag1,unsigned int tag2);
        bool isExistingBond(unsigned int tag1,unsigned int tag2); //this acesses info in m_existing_bonds_list_tag
        virtual void updateImageVectors();
        void checkBoxSize();
        void checkRcut();
        void checkBondType();
        void setGroupOverlap();
        void resizePossibleBondlists();
        void resizeExistingBondList();
        void allocateParticleArrays();

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

        bool m_box_changed;           //!< Flag if box dimensions changed
        bool m_max_N_changed;         //!< Flag if total number of particles changed

    };

namespace detail
{
//! Export the Evaporator to python
void export_DynamicBondUpdater(pybind11::module& m);
} // end namespace detail

} // end namespace azplugins

} // end namespace hoomd

#endif // AZPLUGINS_TYPE_UPDATER_H_
