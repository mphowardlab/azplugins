// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file DynamicBondUpdater.cc
 * \brief Definition of DynamicBondUpdater
 */

#include "DynamicBondUpdater.h"
#include "hoomd/RandomNumbers.h"
#include "RNGIdentifiers.h"
#include "hoomd/md/NeighborListTree.h"

#include <set>
#include <algorithm>

namespace hoomd
{
namespace azplugins
{

/*!
 * \param sysdef System definition
 *
 * The system is initialized in a configuration that will be not forming any bonds.
 * This constructor requires that the user properly initialize the system via setters.
 */
DynamicBondUpdater::DynamicBondUpdater(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<Trigger> trigger,
                                      // std::shared_ptr<md::NeighborList> pair_nlist,
                                       std::shared_ptr<ParticleGroup> group_1,
                                       std::shared_ptr<ParticleGroup> group_2,
                                       uint16_t seed)
          : Updater(sysdef, trigger),
           m_group_1(group_1),
           m_group_2(group_2),
           m_groups_identical(false),
           m_r_cut(0),
           m_probability(0.0),
           m_bond_type(0),
           m_max_bonds_group_1(0),
           m_max_bonds_group_2(0),
           m_seed(seed),
         //  m_pair_nlist(pair_nlist),
           m_pair_nlist_exclusions_set(false),
           m_box_changed(true),
           m_max_N_changed(true)
    {
      m_exec_conf->msg->notice(5) << "Constructing DynamicBondUpdater" << std::endl;

      m_pdata->getBoxChangeSignal().connect<DynamicBondUpdater, &DynamicBondUpdater::slotBoxChanged>(this);
      m_pdata->getGlobalParticleNumberChangeSignal().connect<DynamicBondUpdater, &DynamicBondUpdater::slotNumParticlesChanged>(this);

      m_bond_data = m_sysdef->getBondData();

      m_pair_internal_nlist = std::shared_ptr<hoomd::md::NeighborList>(
        new hoomd::md::NeighborListTree(sysdef, 0.0));
      m_pair_internal_nlist->setStorageMode(hoomd::md::NeighborList::full);

      setGroupOverlap();

      setCutoffs();

      m_max_bonds = 4;
      m_max_bonds_overflow = 0;
      m_num_all_possible_bonds = 0;


    }


DynamicBondUpdater::DynamicBondUpdater(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<Trigger> trigger,
                         //std::shared_ptr<md::NeighborList> pair_nlist,
                         std::shared_ptr<ParticleGroup> group_1,
                         std::shared_ptr<ParticleGroup> group_2,
                         uint16_t seed,
                         const Scalar r_cut,
                         const Scalar probability,
                         unsigned int max_bonds_group_1,
                         unsigned int max_bonds_group_2,
                         unsigned int bond_type
                         )
        : Updater(sysdef, trigger),
         m_group_1(group_1),
         m_group_2(group_2),
         m_groups_identical(false),
         m_r_cut(r_cut),
         m_probability(probability),
         m_bond_type(bond_type),
         m_max_bonds_group_1(max_bonds_group_1),
         m_max_bonds_group_2(max_bonds_group_2),
         m_seed(seed),
       //  m_pair_nlist(pair_nlist),
         m_pair_nlist_exclusions_set(false),
         m_box_changed(true),
         m_max_N_changed(true)
    {

    m_exec_conf->msg->notice(5) << "Constructing DynamicBondUpdater" << std::endl;

    m_pdata->getBoxChangeSignal().connect<DynamicBondUpdater, &DynamicBondUpdater::slotBoxChanged>(this);
    m_pdata->getGlobalParticleNumberChangeSignal().connect<DynamicBondUpdater, &DynamicBondUpdater::slotNumParticlesChanged>(this);

    m_bond_data = m_sysdef->getBondData();

    m_pair_internal_nlist = std::shared_ptr<hoomd::md::NeighborList>(
        new hoomd::md::NeighborListTree(sysdef, 0.0));

    setGroupOverlap();
    m_pair_internal_nlist->setStorageMode(hoomd::md::NeighborList::full);
    setCutoffs();

    m_max_bonds = 4;
    m_max_bonds_overflow = 0;
    m_num_all_possible_bonds = 0;


    }

DynamicBondUpdater::~DynamicBondUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying DynamicBondUpdater" << std::endl;

    m_pdata->getBoxChangeSignal().disconnect<DynamicBondUpdater, &DynamicBondUpdater::slotBoxChanged>(this);
    m_pdata->getGlobalParticleNumberChangeSignal().disconnect<DynamicBondUpdater, &DynamicBondUpdater::slotNumParticlesChanged>(this);

    }

/*!
* \param timestep Timestep update is called
*/
void DynamicBondUpdater::update(uint64_t timestep)
    {
      // don't do anything if either one of the groups is  empty
      if (m_group_1->getNumMembers() == 0 || m_group_2->getNumMembers() == 0)
      return;
      // don't do anything if maximum number of bonds is zero
      if (m_max_bonds_group_1 == 0 || m_max_bonds_group_2 == 0)
      return;

      // update properties that depend on the box
      if (m_box_changed)
      {
        checkBoxSize();
        m_box_changed = false;
      }

      // update properties that depend on the number of particles
      if (m_max_N_changed)
      {
        allocateParticleArrays();
        m_max_N_changed = false;
      }

      { // in its own scope such that the neighbor list can be accsessed in filterPossibleBonds()

      // rebuild the list of possible bonds to the size of the maximum number of neighbors
      // any particle currently has
      m_pair_internal_nlist->compute(timestep);

      //find maximum number of neighbors any particle in group one has
      ArrayHandle<unsigned int> h_n_neigh(m_pair_internal_nlist->getNNeighArray(), access_location::host, access_mode::read);
      for (unsigned int group_idx = 0; group_idx < m_group_1->getNumMembers(); group_idx++)
        {
          unsigned int i = m_group_1->getMemberIndex(group_idx);
          const unsigned int n_neigh = h_n_neigh.data[i];
          if (n_neigh > m_max_bonds_overflow)
            m_max_bonds_overflow = n_neigh;
        }
        //if more neighbors than m_max_bonds, resize the list that saves the possible bonds for all particles
        // if we overflowed, need to reallocate memory and re-traverse the neighbor list
        if (m_max_bonds < m_max_bonds_overflow)
        {
          resizePossibleBondlists();
        }
      }

      filterPossibleBonds();
      // this function is not easily implemented on the GPU, uses addBondedGroup()
      makeBonds(timestep);

    }

// todo: should go into helper class/separate file?
// bonds need to be sorted such that dublicates end up next to each other, otherwise
// unique will not work properly. If the  bond length of the potential bond is different, we can
// sort according to that, but there might be the case where multiple possible bond lengths are exactly identical,
// e.g. particles on a lattice.
// This is hiracical sorting: first according to possible bond distance r_ab_sq, then after first tag_a, last after second tag_b.
// Should work given that the tags are oredered within each pair, (tag_a,tag_b,d_ab_sq) with tag_a < tag_b.

// todo: because it's possible uniquely map a pair to a single int (and back!) with a pairing function,
// the possible bond array could be restructured into a different data structure?
// if we don't keep the possible bond length, a unsigned int array could hold all information needed
// when would that lead to problems with overflow from 0.5*(tag_a+tag_b)*(tag_a+tag_b+1)+tag_b being too large?

bool SortBonds(Scalar3 i, Scalar3 j)
    {
      const Scalar r_sq_1 = i.z;
      const Scalar r_sq_2 = j.z;
      if (r_sq_1==r_sq_2)
      {
        const unsigned int tag_11 = __scalar_as_int(i.x);
        const unsigned int tag_21 = __scalar_as_int(j.x);
        if (tag_11==tag_21)
        {
          const unsigned int tag_12 = __scalar_as_int(i.y);
          const unsigned int tag_22 = __scalar_as_int(j.y);
          return tag_22>tag_12;
        }
        else
        {
          return tag_21>tag_11;
        }
      }
      else
      {
        return r_sq_2>r_sq_1;
      }
    }

// Cantor paring function can also be used for comparison
bool CompareBonds(Scalar3 i, Scalar3 j)
    {
      const unsigned int tag_11 = __scalar_as_int(i.x);
      const unsigned int tag_12 = __scalar_as_int(i.y);
      const unsigned int tag_21 = __scalar_as_int(j.x);
      const unsigned int tag_22 = __scalar_as_int(j.y);

      if ((tag_11==tag_21 && tag_12==tag_22))
      {
        return true;
      }
      else
      {
        return false;
      }
    }


bool DynamicBondUpdater::CheckisExistingLegalBond(Scalar3 i)
    {
      const unsigned int tag_1 = (unsigned int)__scalar_as_int(i.x);
      const unsigned int tag_2 = (unsigned int)__scalar_as_int(i.y);
      // (0,0,0.0) is the default "empty" value - todo: have a "invalid" unsigned int ? how to fill/reset with memset()?
      if (tag_1==0 && tag_2==0 )
      {
        return true;
      }
      else
      {
        return isExistingBond(tag_1,tag_2);
      }
    }

void DynamicBondUpdater::calculateExistingBonds()
    {

      // reset exisitng bond lists to zero
      m_n_existing_bonds.zeroFill();
      m_existing_bonds_list.zeroFill();

      ArrayHandle<typename BondData::members_t> h_bonds(m_bond_data->getMembersArray(), access_location::host, access_mode::read);

      // for each of the bonds in the system - regardless of their type
      const unsigned int size = (unsigned int)m_bond_data->getN();
      for (unsigned int i = 0; i < size; i++)
      {
        // lookup the tag of each of the particles participating in the bond
        const typename BondData::members_t& bond = h_bonds.data[i];
        unsigned int tag1 = bond.tag[0];
        unsigned int tag2 = bond.tag[1];

        assert(tag1 <= m_pdata->getMaximumTag());
        assert(tag2 <= m_pdata->getMaximumTag());

        bool overflowed = false;

        ArrayHandle<unsigned int> h_n_existing_bonds(m_n_existing_bonds, access_location::host, access_mode::readwrite);
        // resize the list if necessary
        if (h_n_existing_bonds.data[tag1] == m_existing_bonds_list_indexer.getH())
        overflowed = true;
        if (h_n_existing_bonds.data[tag2] == m_existing_bonds_list_indexer.getH())
        overflowed = true;

        if (overflowed) resizeExistingBondList();

        { // explicit scoping such that resizeExistingBondList can resize the m_existing_bonds_list array before this
         ArrayHandle<unsigned int> h_existing_bonds_list(m_existing_bonds_list, access_location::host, access_mode::readwrite);
        // add tag_b to tag_a's existing bonds list
        unsigned int pos_a = h_n_existing_bonds.data[tag1];
        assert(pos_a < m_existing_bonds_list_indexer.getH());
        h_existing_bonds_list.data[m_existing_bonds_list_indexer(tag1,pos_a)] = tag2;
        h_n_existing_bonds.data[tag1]++;

        // add tag_a to tag_b's existing bonds list
        unsigned int pos_b = h_n_existing_bonds.data[tag2];
        assert(pos_b < m_existing_bonds_list_indexer.getH());
        h_existing_bonds_list.data[m_existing_bonds_list_indexer(tag2,pos_b)] = tag1;
        h_n_existing_bonds.data[tag2]++;

        }

      }

    }

/*! \param tag1 First particle tag in the pair
    \param tag2 Second particle tag in the pair
    \return true if the particles \a tag1 and \a tag2 are bonded
*/
bool DynamicBondUpdater::isExistingBond(unsigned int tag1, unsigned int tag2)
    {
    {
      ArrayHandle<unsigned int> h_n_existing_bonds(m_n_existing_bonds, access_location::host, access_mode::read);
      ArrayHandle<unsigned int> h_existing_bonds_list(m_existing_bonds_list, access_location::host, access_mode::read);

      unsigned int n_existing_bonds = (unsigned int)h_n_existing_bonds.data[tag1];
      for (unsigned int i = 0; i < n_existing_bonds; i++)
      {
        if (h_existing_bonds_list.data[m_existing_bonds_list_indexer(tag1,i)] == tag2)
        return true;
      }
      return false;
    }
    }


// grows the existing bonds list and its indexer when needed
void DynamicBondUpdater::resizeExistingBondList()
    {
      unsigned int new_height = m_existing_bonds_list_indexer.getH() + 1;
      m_existing_bonds_list.resize(m_pdata->getMaxN(), new_height);
      // update the indexer
      m_existing_bonds_list_indexer = Index2D((unsigned int)m_existing_bonds_list.getPitch(), new_height);

      m_exec_conf->msg->notice(6) << "DynamicBondUpdater: (Re-)size existing bond list, new size " << new_height << " bonds per particle " << std::endl;
    }

// grows the all possible bonds list when needed in increments of 4, inspired by the neighbor list
void DynamicBondUpdater::resizePossibleBondlists()
    {
      // round up to nearest multiple of 4
      m_max_bonds_overflow = (m_max_bonds_overflow > 4) ? (m_max_bonds_overflow + 3) & ~3 : 4;
      m_max_bonds = m_max_bonds_overflow;
      m_max_bonds_overflow = 0;
      unsigned int size = m_group_1->getNumMembers()*m_max_bonds;

      m_all_possible_bonds.resize(size);
      m_all_possible_bonds.zeroFill();
      m_num_all_possible_bonds=0;

      m_exec_conf->msg->notice(6) << "DynamicBondUpdater: (Re-)size possible bond list, new size " << m_max_bonds << " bonds per particle " << std::endl;

    }



// allocates all arrays depending on the particles and groups
void DynamicBondUpdater::allocateParticleArrays()
    {

      { // explicit scoping so that calculateExistingBonds can accsess the arrays after
      GPUArray<Scalar3> all_possible_bonds(m_group_1->getNumMembers()*m_max_bonds, m_exec_conf);
      m_all_possible_bonds.swap(all_possible_bonds);
      m_all_possible_bonds.zeroFill();

      GPUArray<unsigned int> n_existing_bonds(m_pdata->getMaxN(), m_exec_conf);
      m_n_existing_bonds.swap(n_existing_bonds);
      m_n_existing_bonds.zeroFill();

      GPUArray<unsigned int> existing_bonds_list(m_pdata->getMaxN(),1, m_exec_conf);
      m_existing_bonds_list.swap(existing_bonds_list);
      m_existing_bonds_list.zeroFill();

      m_existing_bonds_list_indexer = Index2D((unsigned int)m_existing_bonds_list.getPitch(), (unsigned int)m_existing_bonds_list.getHeight());
      }
      calculateExistingBonds();
    }



/*! This function takes the information about neighbors between group_2 and group_1 saved in m_nlist and
* m_n_neigh and copies pairs within the m_r_cut cutoff distance into the m_all_possible_bonds array.
* Then, all invalid (0,0,0), dublicated, and existing bonds  are removed from m_all_possible_bonds. It
* is sorted by distance  (shortest to longest) between the two particles in the possible bond.
*/
void DynamicBondUpdater::filterPossibleBonds()
    {

      //copy data from neighbor list to h_all_possible_bonds
      ArrayHandle<unsigned int> h_nlist(m_pair_internal_nlist->getNListArray(), access_location::host, access_mode::read);
      ArrayHandle<unsigned int> h_n_neigh(m_pair_internal_nlist->getNNeighArray(), access_location::host, access_mode::read);
      ArrayHandle<size_t> h_n_head_list(m_pair_internal_nlist->getHeadList(), access_location::host, access_mode::read);

      ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
      ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

      // reset the array of possible bonds so it can be re-populated below
      m_all_possible_bonds.zeroFill();

      const BoxDim& box = m_pdata->getBox();

      ArrayHandle<Scalar3> h_all_possible_bonds(m_all_possible_bonds, access_location::host, access_mode::readwrite);

      // Loop over all particles in group 1
      for (unsigned int group_idx = 0; group_idx < m_group_1->getNumMembers(); group_idx++)
      {

        unsigned int i = m_group_1->getMemberIndex(group_idx);
        const unsigned int tag_i = h_tag.data[i];
        const Scalar4 postype_i = h_postype.data[i];

        unsigned int n_curr_bond = 0;
        const Scalar r_cutsq = m_r_cut*m_r_cut;

        const unsigned int n_neigh = (unsigned int)h_n_neigh.data[i];
        const size_t head = h_n_head_list.data[i];

        // loop over all neighbors of this particle
        for (unsigned int l=0; l<n_neigh; ++l)
        {

          // get index of neighbor from neigh_list
          const unsigned int j = h_nlist.data[head + l];
          const unsigned int tag_j = h_tag.data[j];
          Scalar4 postype_j = h_postype.data[j];

          if (m_group_2->isMember(j))
          {

            Scalar3 drij = make_scalar3(postype_j.x,postype_j.y,postype_j.z)
            - make_scalar3(postype_i.x,postype_i.y,postype_i.z);

            // apply periodic boundary conditions
            drij = box.minImage(drij);
            Scalar dr_sq = dot(drij,drij);

            if (dr_sq < r_cutsq)
            {
              if (n_curr_bond < m_max_bonds)
              {
                Scalar3 d;
                if(m_groups_identical)
                {
                  // sort the two tags in this possible bond pair if groups identical
                  const unsigned int tag_a = tag_j > tag_i ? tag_i : tag_j;
                  const unsigned int tag_b = tag_j > tag_i ? tag_j : tag_i;

                  d = make_scalar3(__int_as_scalar(tag_a),__int_as_scalar(tag_b),dr_sq);
                }
                else
                {
                  d = make_scalar3(__int_as_scalar(tag_i),__int_as_scalar(tag_j),dr_sq);
                }

                h_all_possible_bonds.data[group_idx*m_max_bonds + n_curr_bond] = d;
              }
              ++n_curr_bond;
            }
          }
        }
      }


      {
      //now sort and select down
      m_num_all_possible_bonds = 0;
      unsigned int size = m_group_1->getNumMembers()*m_max_bonds;

      // remove a possible bond if it already exists. It also removes zeros, e.g.
      // (0,0,0), which fill the unused spots in the array.
      auto last2 = std::remove_if(h_all_possible_bonds.data,
                   h_all_possible_bonds.data + size,
                   [this](Scalar3 i) {return CheckisExistingLegalBond(i); });

      m_num_all_possible_bonds = (unsigned int) std::distance(h_all_possible_bonds.data,last2);

      // then sort array by distance between particles in the found possible bond pairs
      // performance is better if remove_if happens before sort
      std::sort(h_all_possible_bonds.data, h_all_possible_bonds.data + m_num_all_possible_bonds, SortBonds);

      // now make sure each possible bond is in the array only once by comparing tags
      auto last = std::unique(h_all_possible_bonds.data, h_all_possible_bonds.data + m_num_all_possible_bonds, CompareBonds);
      m_num_all_possible_bonds = (unsigned int)std::distance(h_all_possible_bonds.data,last);

      // at this point, the sub-array: h_all_possible_bonds[0,m_num_all_possible_bonds]
      // should contain only unique entries of possible bonds which are not yet formed.

    }

  }

/*! This function actually creates the bonds by looping over the entries in m_all_possible_bonds
*  and adding them to the system (m_bond_data->addBondedGroup), to the existing bonds, as well as
*  to the neighbor list used by the rest of the simulation if the exclusions of that neighbor list
*  should be updated.
*
* Note: this function is very hard to parallelize on the GPU since we need to go through the bonds sequentially
* to prevent forming too many bonds in one step. Have not found a good way of doing this on the GPU.
*/
void DynamicBondUpdater::makeBonds(uint64_t timestep)
  {

    ArrayHandle<Scalar3> h_all_possible_bonds(m_all_possible_bonds, access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_n_existing_bonds(m_n_existing_bonds, access_location::host, access_mode::readwrite);


    // we need to count how many bonds are in the h_all_possible_bonds array for a given tag
    // so that we don't end up forming too many bonds in one step. "AddtoExistingBonds" increases the count in
    // h_n_existing_bonds in the for loop below as we go, so no extra bookkeeping should be needed.
    // This also makes it very difficult to do on the GPU.

    //todo: can this for loop be simplified/parallelized?
    for (unsigned int i = 0; i < m_num_all_possible_bonds; i++)
    {
      Scalar3 d = h_all_possible_bonds.data[i];

      unsigned int tag_i = (unsigned int)__scalar_as_int(d.x);
      unsigned int tag_j = (unsigned int)__scalar_as_int(d.y);

     //todo: put in other external criteria here, e.g. max number of bonds possible in one step, etc.
     //todo: randomize which bonds are formed or keep them ordered by their distances ?
     //todo: would it be faster/better to create the rng outside of the loop?
    hoomd::RandomGenerator rng(
                hoomd::Seed(hoomd::azplugins::detail::RNGIdentifier::DynamicBondUpdater,
                            timestep,
                            m_seed),
                hoomd::Counter());

     hoomd::UniformDistribution<Scalar> uniform(0, 1);
     const Scalar random = uniform(rng);

      if ((m_max_bonds_group_1 > h_n_existing_bonds.data[tag_i]) &&
          (m_max_bonds_group_2 > h_n_existing_bonds.data[tag_j]) &&
          (random < m_probability))
        {
          m_bond_data->addBondedGroup(Bond(m_bond_type,tag_i,tag_j));

          // BEGIN DynamicBondUpdater::AddtoExistingBonds
          assert(tag_i <= m_pdata->getMaximumTag());
          assert(tag_j <= m_pdata->getMaximumTag());

          bool overflowed = false;

          // resize the list if necessary
          if (h_n_existing_bonds.data[tag_i] == m_existing_bonds_list_indexer.getH())
          overflowed = true;
          if (h_n_existing_bonds.data[tag_j] == m_existing_bonds_list_indexer.getH())
          overflowed = true;

          if (overflowed) resizeExistingBondList();

          {// explicit scoping such that resizeExistingBondList can resize the m_existing_bonds_list array before this
          ArrayHandle<unsigned int> h_existing_bonds_list(m_existing_bonds_list, access_location::host, access_mode::readwrite);

          // add tag_b to tag_a's existing bonds list
          unsigned int pos_a = h_n_existing_bonds.data[tag_i];
          assert(pos_a < m_existing_bonds_list_indexer.getH());
          h_existing_bonds_list.data[m_existing_bonds_list_indexer(tag_i,pos_a)] = tag_j;
          h_n_existing_bonds.data[tag_i]++;

          // add tag_a to tag_b's existing bonds list
          unsigned int pos_b = h_n_existing_bonds.data[tag_j];
          assert(pos_b < m_existing_bonds_list_indexer.getH());
          h_existing_bonds_list.data[m_existing_bonds_list_indexer(tag_j,pos_b)] = tag_i;
          h_n_existing_bonds.data[tag_j]++;

          }
          if (m_pair_nlist_exclusions_set)
            {
            m_pair_nlist -> addExclusion(tag_i,tag_j);
            }
          // the internal neigh list should always get the exclusions updates to not re-find already existing bonds over and over
          m_pair_internal_nlist -> addExclusion(tag_i,tag_j);
        }
      }

  }


/*!
* Check that the largest neighbor search radius is not bigger than twice the shortest box size.
* Raises an error if this condition is not met.
*/
void DynamicBondUpdater::checkBoxSize()
    {
      const BoxDim& box = m_pdata->getBox();
      const uchar3 periodic = box.getPeriodic();

      // check that rcut fits in the box
      Scalar3 nearest_plane_distance = box.getNearestPlaneDistance();
      Scalar rmax = m_r_cut;

      if ((periodic.x && nearest_plane_distance.x <= rmax * 2.0) ||
      (periodic.y && nearest_plane_distance.y <= rmax * 2.0) ||
      (m_sysdef->getNDimensions() == 3 && periodic.z && nearest_plane_distance.z <= rmax * 2.0))
      {
        m_exec_conf->msg->error() << "DynamicBondUpdater: Simulation box is too small! Particles would be interacting with themselves." << std::endl;
        throw std::runtime_error("Error in DynamicBondUpdater, Simulation box too small.");
      }
    }

/*! Calculates if the two groups have overlap or not. Returns an error if  partial
* overlap is detected.
*/
void DynamicBondUpdater::setGroupOverlap()
    {
      if(m_group_1->getNumMembers()==0)
      {
        m_exec_conf->msg->warning() << "DynamicBondUpdater: First group group_1 appears to be empty. No bonds will be formed. " << std::endl;
      }

      if(m_group_2->getNumMembers()==0)
      {
        m_exec_conf->msg->warning() << "DynamicBondUpdater: Second group group_2 appears to be empty. No bonds will be formed. " << std::endl;
      }
      {
      //check if the two groups are either identical or have no overlap
      ArrayHandle<unsigned int> h_index_group_1(m_group_1->getIndexArray(), access_location::host, access_mode::read);

      // count particles which are in both groups. Should be either zero of them or all of them.
      unsigned int overlap = 0;
      for (unsigned int i=0; i<m_group_1->getNumMembers(); ++i)
      {
        unsigned int idx = h_index_group_1.data[i];
        if (m_group_2->isMember(idx))
        overlap++;
      }

      if( overlap>0 && overlap != m_group_1->getNumMembers())
      {
        m_exec_conf->msg->error() << "DynamicBondUpdater: group 1 and group 2 have " << overlap  << " overlaps. Partially overlapping groups are not implemented." << std::endl;
        throw std::runtime_error("Partial overlapping groups in DynamicBondUpdater");
      }

      if(overlap==m_group_1->getNumMembers())
      {
        m_groups_identical=true;
      }
      }
    }

/*! Sets cutoffs based on types present in the two groups to save some performance from
* the neighbor list.
*/
void DynamicBondUpdater::setCutoffs()
    {
    {

    // set all rcuts to zero first, then only set the one between group1 and group 2 particle types to be m_r_cut
     unsigned int NTypes = m_pdata -> getNTypes();
      for (unsigned int i=0; i< NTypes; ++i)
      {
        for (unsigned int j=0; j< NTypes; ++j)
          {
            m_pair_internal_nlist->setRcut(i,j,0);
          }
      }

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::read);


    ArrayHandle<unsigned int> h_index_group_1(m_group_1->getIndexArray(), access_location::host, access_mode::read);

    // finding all types in group 1
    std::set<unsigned int> types_group_1;
    for (unsigned int i=0; i<m_group_1->getNumMembers(); ++i)
      {
        unsigned int idx = h_index_group_1.data[i];
        Scalar4 type = h_pos.data[idx];
        types_group_1.insert(__scalar_as_int(type.w));
      }

    std::set<unsigned int> types_group_2;
    if (m_groups_identical == false)
    {
    ArrayHandle<unsigned int> h_index_group_2(m_group_2->getIndexArray(), access_location::host, access_mode::read);

    // finding all types in group 2
    for (unsigned int i=0; i<m_group_2->getNumMembers(); ++i)
      {
        unsigned int idx = h_index_group_2.data[i];
        Scalar4 type =  h_pos.data[idx];
        types_group_2.insert(__scalar_as_int(type.w));
      }
    }
    else
    {
    types_group_2 = types_group_1;
    }

      // looping over types in group 1 and 2 to set the cutoff to m_r_cut
      for (const auto& element_1 : types_group_1)
      {
        for (const auto& element_2 : types_group_2)
        {
          m_pair_internal_nlist->setRcut(element_1,element_2,m_r_cut);
          m_pair_internal_nlist->setRcut(element_2,element_1,m_r_cut);
        }
      }

    }
    }


// Check that the given cutoff value is valid
void DynamicBondUpdater::checkRcut()
    {
      if (m_r_cut <= 0.0)
      {
        m_exec_conf->msg->error() << "DynamicBondUpdater: Requested cutoff distance is less than or equal to zero" << std::endl;
        throw std::runtime_error("Error initializing DynamicBondUpdater");
      }
      checkBoxSize();
    }

void DynamicBondUpdater::checkProbability()
{
      if (m_probability < 0.0)
      {
        m_exec_conf->msg->error() << "DynamicBondUpdater: Requested probability  is less than zero" << std::endl;
        throw std::runtime_error("Error initializing DynamicBondUpdater");
      }
      else if (m_probability > 1.0)
      {
         m_exec_conf->msg->error() << "DynamicBondUpdater: Requested probability is larger than one" << std::endl;
        throw std::runtime_error("Error initializing DynamicBondUpdater");
      }

}

void DynamicBondUpdater::checkMaxBondsGroup()
{
      if (m_max_bonds_group_1 < 0.0 or m_max_bonds_group_2 < 0.0 )
      {
        m_exec_conf->msg->error() << "DynamicBondUpdater: Max number of bonds that groups can form is negative. Check parameters." << std::endl;
        throw std::runtime_error("Error initializing DynamicBondUpdater");
      }
      else if (m_max_bonds_group_1 > 10 or m_max_bonds_group_2 > 10  )
      {
         m_exec_conf->msg->warning() << "DynamicBondUpdater: Requested number of bonds that can form is very large. This can lead to performance issues." << std::endl;
      }

}

// Check that the given bond type is valid
void DynamicBondUpdater::checkBondType()
    {
      if (m_bond_type >= m_bond_data -> getNTypes())
      {
        m_exec_conf->msg->error() << "DynamicBondUpdater: bond type id " << m_bond_type  << " is not a valid bond type." << std::endl;
        throw std::runtime_error("Invalid bond type for DynamicBondUpdater");
      }
    }


namespace detail
{
/*!
* \param m Python module to export to
*/

void export_DynamicBondUpdater(pybind11::module& m)
    {

    pybind11::class_< DynamicBondUpdater, Updater,std::shared_ptr<DynamicBondUpdater>>(
      m,
      "DynamicBondUpdater")
    .def(pybind11::init<std::shared_ptr<SystemDefinition>,
      std::shared_ptr<Trigger>,
      //std::shared_ptr<md::NeighborList>,
      std::shared_ptr<ParticleGroup>,
      std::shared_ptr<ParticleGroup>,
      uint16_t>())
    .def(pybind11::init<std::shared_ptr<SystemDefinition>,
      std::shared_ptr<Trigger>,
      //std::shared_ptr<md::NeighborList>,
      std::shared_ptr<ParticleGroup>,
      std::shared_ptr<ParticleGroup>,
      uint16_t,
      Scalar,
      Scalar,
      unsigned int,
      unsigned int,
      unsigned int>())
      .def_property("r_cut", &DynamicBondUpdater::getRcut, &DynamicBondUpdater::setRcut)
      .def_property("probability", &DynamicBondUpdater::getProbability, &DynamicBondUpdater::setProbability)
      .def_property("bond_type",&DynamicBondUpdater::getBondType, &DynamicBondUpdater::setBondType)
      .def("setNlist",&DynamicBondUpdater::setNeighborList)
      .def_property("max_bonds_group_1", &DynamicBondUpdater::getMaxBondsGroup1, &DynamicBondUpdater::setMaxBondsGroup1)
      .def_property("max_bonds_group_2", &DynamicBondUpdater::getMaxBondsGroup2, &DynamicBondUpdater::setMaxBondsGroup2);
    }
} // end namespace detail

} // end namespace azplugins

} // end namespace hoomd