// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file DynamicBondUpdater.cc
 * \brief Definition of DynamicBondUpdater
 */

#include "DynamicBondUpdater.h"


namespace azplugins
{

DynamicBondUpdater::DynamicBondUpdater(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<NeighborList> nlist,
                         bool nlist_exclusions_set,
                         std::shared_ptr<ParticleGroup> group_1,
                         std::shared_ptr<ParticleGroup> group_2,
                         const Scalar r_cut,
                         const Scalar r_buff,
                         unsigned int bond_type,
                         unsigned int max_bonds_group_1,
                         unsigned int max_bonds_group_2)
        : Updater(sysdef), m_nlist(nlist), m_nlist_exclusions_set(nlist_exclusions_set), m_group_1(group_1), m_group_2(group_2), m_r_cut(r_cut),m_r_buff(r_buff),
         m_bond_type(bond_type),m_max_bonds_group_1(max_bonds_group_1),m_max_bonds_group_2(max_bonds_group_2),
         m_box_changed(true), m_max_N_changed(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing DynamicBondUpdater" << std::endl;

    if (m_r_cut < 0.0 || m_r_buff < 0.0)
        {
        m_exec_conf->msg->error() << "DynamicBondUpdater: Requested cutoff distance or buffer radius is less than zero" << std::endl;
        throw std::runtime_error("Error initializing DynamicBondUpdater");
        }

    m_pdata->getBoxChangeSignal().connect<DynamicBondUpdater, &DynamicBondUpdater::slotBoxChanged>(this);
    m_pdata->getGlobalParticleNumberChangeSignal().connect<DynamicBondUpdater, &DynamicBondUpdater::slotNumParticlesChanged>(this);
    m_pdata->getParticleSortSignal().connect<DynamicBondUpdater, &DynamicBondUpdater::slotParticlesSort>(this);


    m_bond_data = m_sysdef->getBondData();

    m_max_bonds = 4;
    m_max_bonds_overflow = 0;
    m_num_all_possible_bonds=0;

    // allocate initial Memory - grows if necessary
    GPUArray<Scalar3> all_possible_bonds(m_group_1->getNumMembers()*m_max_bonds, m_exec_conf);
    m_all_possible_bonds.swap(all_possible_bonds);

    // todo: reset if group sizes changes?
    // if groups change during the simulation this updater might just not work properly - groups don't have a change signal?
    // can getNumTypesChangeSignal() be used as a proxy?
    m_aabbs.resize(m_group_2->getNumMembers());

    checkSystemSetup();

    }

DynamicBondUpdater::~DynamicBondUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying DynamicBondUpdater" << std::endl;

    m_pdata->getBoxChangeSignal().disconnect<DynamicBondUpdater, &DynamicBondUpdater::slotBoxChanged>(this);
    m_pdata->getGlobalParticleNumberChangeSignal().disconnect<DynamicBondUpdater, &DynamicBondUpdater::slotNumParticlesChanged>(this);
    m_pdata->getParticleSortSignal().disconnect<DynamicBondUpdater, &DynamicBondUpdater::slotParticlesSort>(this);

    }


/*!
 * \param timestep Timestep update is called
 */
void DynamicBondUpdater::update(unsigned int timestep)
{
   if (m_prof) m_prof->push("DynamicBondUpdater");

    // don't do anything if either one of the groups is  empty
    const unsigned int group_size_1 = m_group_1->getNumMembers();
    const unsigned int group_size_2 = m_group_2->getNumMembers();
    if (group_size_1 == 0 || group_size_2 == 0)
        return;

    // update properties that depend on the box
    if (m_box_changed)
        {
        checkBoxSize();
        updateImageVectors();
        m_box_changed = false;
        }

    // update properties that depend on the number of particles
    if (m_max_N_changed)
        {
        allocateParticleArrays();
        m_max_N_changed = false;
        }

    // rebuild the list of possible bonds until there is no overflow
    bool overflowed = false;

    if (needsUpdating())
    {
    buildTree();
    do
        {
        traverseTree();
        overflowed = m_max_bonds < m_max_bonds_overflow;
        // if we overflowed, need to reallocate memory and re-traverse the tree
        if (overflowed)
            {
            resizePossibleBondlists();
            }
        } while (overflowed);
    setLastUpdatedPos();
    }

    filterPossibleBonds();
    // this function is not easily implemented on the GPU, uses addBondedGroup()
    makeBonds();
    if (m_prof) m_prof->pop();

}

// todo: should go into helper class/separate file?
// bonds need to be sorted such that dublicates end up next to each other, otherwise
// unique will not work properly. If the possible bond length is different, we can
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

// todo: migrate to separate file/class?
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

//todo: find a better descriptive name for this function
bool DynamicBondUpdater::CheckisExistingLegalBond(Scalar3 i)
{
  const unsigned int tag_1 = __scalar_as_int(i.x);
  const unsigned int tag_2 = __scalar_as_int(i.y);

  // (0,0,0.0) is the default "empty" value - todo: have a "invalid" unsigned int ? how to fill/reset with memset()?
  if (tag_1==0 && tag_2==0 ){
     return true;
   }else{
     return isExistingBond(tag_1,tag_2);
   }

}

void DynamicBondUpdater::calculateExistingBonds()
{

  // reset exisitng bond list
  ArrayHandle<unsigned int> h_n_existing_bonds(m_n_existing_bonds, access_location::host, access_mode::overwrite);
  memset((void*)h_n_existing_bonds.data,0,sizeof(unsigned int)*m_pdata->getRTags().size());

  ArrayHandle<unsigned int> h_existing_bonds_list(m_existing_bonds_list, access_location::host, access_mode::overwrite);
  memset((void*)h_existing_bonds_list.data,0,sizeof(unsigned int)*m_pdata->getRTags().size()*m_existing_bonds_list_indexer.getH());

  ArrayHandle<typename BondData::members_t> h_bonds(m_bond_data->getMembersArray(), access_location::host, access_mode::read);

  // for each of the bonds
  const unsigned int size = (unsigned int)m_bond_data->getN();
  for (unsigned int i = 0; i < size; i++)
  {
      // lookup the tag of each of the particles participating in the bond
      const typename BondData::members_t& bond = h_bonds.data[i];
      unsigned int tag1 = bond.tag[0];
      unsigned int tag2 = bond.tag[1];

    //  keep track of all bond types in the system - does this make sense?
    //  if (type == m_bond_type)
    //  {
        AddtoExistingBonds(tag1,tag2);
    //  }
  }

}

/*! \param tag1 First particle tag in the pair
    \param tag2 Second particle tag in the pair
    \return true if the particles \a tag1 and \a tag2 are bonded
*/
bool DynamicBondUpdater::isExistingBond(unsigned int tag1, unsigned int tag2)
{
    ArrayHandle<unsigned int> h_n_existing_bonds(m_n_existing_bonds, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_existing_bonds_list(m_existing_bonds_list, access_location::host, access_mode::read);
    unsigned int n_existing_bonds = h_n_existing_bonds.data[tag1];

    for (unsigned int i = 0; i < n_existing_bonds; i++)
        {
        if (h_existing_bonds_list.data[m_existing_bonds_list_indexer(tag1,i)] == tag2)
            return true;
        }
    return false;
    }



void DynamicBondUpdater::AddtoExistingBonds(unsigned int tag1,unsigned int tag2)
{
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

  ArrayHandle<unsigned int> h_existing_bonds_list(m_existing_bonds_list, access_location::host, access_mode::readwrite);

  // add tag2 to tag1's existing bonds list
  unsigned int pos1 = h_n_existing_bonds.data[tag1];
  assert(pos1 < m_existing_bonds_list_indexer.getH());
  h_existing_bonds_list.data[m_existing_bonds_list_indexer(tag1,pos1)] = tag2;
  h_n_existing_bonds.data[tag1]++;

  // add tag1 to tag2's existing bonds list
  unsigned int pos2 = h_n_existing_bonds.data[tag2];
  assert(pos2 < m_existing_bonds_list_indexer.getH());
  h_existing_bonds_list.data[m_existing_bonds_list_indexer(tag2,pos2)] = tag1;
  h_n_existing_bonds.data[tag2]++;
}

//todo: should the list be grown more than 1 at a time for efficiency?
void DynamicBondUpdater::resizeExistingBondList()
    {
    unsigned int new_height = m_existing_bonds_list_indexer.getH() + 1;
    m_existing_bonds_list.resize(m_pdata->getRTags().size(), new_height);
    // update the indexer
    m_existing_bonds_list_indexer = Index2D(m_existing_bonds_list.getPitch(), new_height);
    m_exec_conf->msg->notice(6) << "DynamicBondUpdater: (Re-)size existing bond list, new size " << new_height << " bonds per particle " << std::endl;

    //do we need to recalculate existing bonds when resizing?

    }


void DynamicBondUpdater::resizePossibleBondlists()
    {
      // round up to nearest multiple of 4
      m_max_bonds_overflow = (m_max_bonds_overflow > 4) ? (m_max_bonds_overflow + 3) & ~3 : 4;
      m_max_bonds = m_max_bonds_overflow;
      m_max_bonds_overflow = 0;
      unsigned int size = m_group_1->getNumMembers()*m_max_bonds;
      m_all_possible_bonds.resize(size);
      m_num_all_possible_bonds=0;

      GlobalArray<unsigned int> nlist(m_max_bonds*m_pdata->getMaxN(), m_exec_conf);
      m_n_list.swap(nlist);

      m_exec_conf->msg->notice(6) << "DynamicBondUpdater: (Re-)size possible bond list, new size " << m_max_bonds << " bonds per particle " << std::endl;

     forceUpdate();
    }


void DynamicBondUpdater::allocateParticleArrays()
  {

    GPUArray<unsigned int> n_existing_bonds(m_pdata->getRTags().size(), m_exec_conf);
    m_n_existing_bonds.swap(n_existing_bonds);

    GPUArray<unsigned int> existing_bonds_list(m_pdata->getRTags().size(),1, m_exec_conf);
    m_existing_bonds_list.swap(existing_bonds_list);
    m_existing_bonds_list_indexer = Index2D(m_existing_bonds_list.getPitch(), m_existing_bonds_list.getHeight());

    ArrayHandle<unsigned int> h_n_existing_bonds(m_n_existing_bonds, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_existing_bonds_list(m_existing_bonds_list, access_location::host, access_mode::overwrite);

    memset(h_n_existing_bonds.data, 0, sizeof(unsigned int)*m_n_existing_bonds.getNumElements());
    memset(h_existing_bonds_list.data, 0, sizeof(unsigned int)*m_existing_bonds_list.getNumElements());

    // allocate m_last_pos
    GlobalArray<Scalar4> last_pos(m_pdata->getMaxN(), m_exec_conf);
    m_last_pos.swap(last_pos);

    // allocate the number of neighbors (per particle)
    GlobalArray<unsigned int> n_neigh(m_pdata->getMaxN(), m_exec_conf);
    m_n_neigh.swap(n_neigh);
    ArrayHandle<unsigned int> h_n_neigh(m_n_neigh, access_location::host, access_mode::overwrite);
    memset(h_n_neigh.data, 0, sizeof(unsigned int)*m_n_neigh.getNumElements());

    // default allocation of m_max_bonds neighbors per particle for the neighborlist
    GlobalArray<unsigned int> nlist(m_max_bonds*m_pdata->getMaxN(), m_exec_conf);
    m_n_list.swap(nlist);

   calculateExistingBonds();
   forceUpdate();

  }

bool DynamicBondUpdater::needsUpdating()
{

  if (m_force_update == true)
  {
    m_force_update = false;
    return true;
  }

  if (m_r_buff < 1e-6) return true;

 //distance checking between last positions (at time of last update) and current positions
 // todo: only check distances between particles in group_1 and 2, not all of them 
 {
   ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

   // sanity check
   assert(h_pos.data);

   // profile
   if (m_prof) m_prof->push("distcheck");

   // temporary storage for the result
   bool result = false;

   // get a local copy of the simulation box too
   const BoxDim& box = m_pdata->getBox();

   // get current nearest plane distances
   Scalar3 L_g = m_pdata->getGlobalBox().getNearestPlaneDistance();

   // Find direction of maximum box length contraction (smallest eigenvalue of deformation tensor)
   Scalar3 lambda = L_g / m_last_L;
   Scalar lambda_min = (lambda.x < lambda.y) ? lambda.x : lambda.y;
   lambda_min = (lambda_min < lambda.z) ? lambda_min : lambda.z;

   ArrayHandle<Scalar4> h_last_pos(m_last_pos, access_location::host, access_mode::read);
   //ArrayHandle<Scalar> h_rcut_max(m_rcut_max, access_location::host, access_mode::read);

   for (unsigned int i = 0; i < m_pdata->getN(); i++)
       {

       // minimum distance within which all particles should be included
       Scalar old_rmin = m_r_cut;

       // maximum value we have checked for neighbors, defined by the buffer layer
       Scalar rmax = old_rmin + m_r_buff;

       // max displacement for each particle (after subtraction of homogeneous dilations)
       const Scalar delta_max = (rmax*lambda_min - old_rmin)/Scalar(2.0);
       Scalar maxsq = (delta_max > 0) ? delta_max*delta_max : 0;

       Scalar3 dx = make_scalar3(h_pos.data[i].x - lambda.x*h_last_pos.data[i].x,
                                 h_pos.data[i].y - lambda.y*h_last_pos.data[i].y,
                                 h_pos.data[i].z - lambda.z*h_last_pos.data[i].z);

       dx = box.minImage(dx);

       if (dot(dx, dx) >= maxsq)
           {
           result = true;
           break;
           }
       }

   #ifdef ENABLE_MPI
   if (m_pdata->getDomainDecomposition())
       {
       if (m_prof) m_prof->push("MPI allreduce");
       // check if migrate criterion is fulfilled on any rank
       int local_result = result ? 1 : 0;
       int global_result = 0;
       MPI_Allreduce(&local_result,
           &global_result,
           1,
           MPI_INT,
           MPI_MAX,
           m_exec_conf->getMPICommunicator());
       result = (global_result > 0);
       if (m_prof) m_prof->pop();
       }
   #endif

   if (m_prof) m_prof->pop();

   return result;
 }

}

/*! Copies the current positions of all particles over to m_last_x etc...
*/
void DynamicBondUpdater::setLastUpdatedPos()
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    // sanity check
    assert(h_pos.data);

    // profile
    if (m_prof) m_prof->push("updatePos");

    // update the last position arrays
    ArrayHandle<Scalar4> h_last_pos(m_last_pos, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        h_last_pos.data[i] = make_scalar4(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z, Scalar(0.0));
        }

    // update last box nearest plane distance
    m_last_L = m_pdata->getGlobalBox().getNearestPlaneDistance();
    m_last_L_local = m_pdata->getBox().getNearestPlaneDistance();

    if (m_prof) m_prof->pop();
    }


// this is based on the NeighborListTree c++ implementation
void DynamicBondUpdater::buildTree()
  {
    if (m_prof) m_prof->push("buildTree");
    //todo: is it worth it to check if rebuild is necessary similar to neighbor list with keeping track of old positions?
    // make tree for group 2
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<hpmc::detail::AABB> h_aabbs(m_aabbs, access_location::host, access_mode::readwrite);

    unsigned int group_size_2 = m_group_2->getNumMembers();

    for (unsigned int group_idx = 0; group_idx < group_size_2; group_idx++)
        {
        unsigned int i = m_group_2->getMemberIndex(group_idx);
        // make a point particle AABB
        vec3<Scalar> my_pos(h_postype.data[i]);
        h_aabbs.data[group_idx] = hpmc::detail::AABB(my_pos,i);
        }

    m_aabb_tree.buildTree(&(h_aabbs.data[0]) , group_size_2);

  if (m_prof) m_prof->pop();
 }


 // this is based on the NeighborListTree c++ implementation
 void DynamicBondUpdater::traverseTree()
   {
       if (m_prof) m_prof->push("traverseTree");

       ArrayHandle<unsigned int> h_nlist(m_n_list, access_location::host, access_mode::overwrite);
       ArrayHandle<unsigned int> h_n_neigh(m_n_neigh, access_location::host, access_mode::overwrite);

       // clear the neighbor counts
       memset(h_n_neigh.data, 0, sizeof(unsigned int)*m_pdata->getMaxN());
       memset(h_nlist.data,0, sizeof(unsigned int)*m_max_bonds*m_pdata->getMaxN());

      // reset content of possible bond list
      ArrayHandle<Scalar3> h_all_possible_bonds(m_all_possible_bonds, access_location::host, access_mode::overwrite);
      const unsigned int size = m_group_1->getNumMembers()*m_max_bonds;
      memset((void*)h_all_possible_bonds.data, 0, sizeof(Scalar3)*size);
      const Scalar r_cutsq = (m_r_cut+m_r_buff)*(m_r_cut+m_r_buff);

      ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
      ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

      // traverse the tree
      // Loop over all particles in group 1
      unsigned int group_size_1 = m_group_1->getNumMembers();
      for (unsigned int group_idx = 0; group_idx < group_size_1; group_idx++)
          {
          unsigned int i = m_group_1->getMemberIndex(group_idx);
          const Scalar4 postype_i = h_postype.data[i];
          const vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

          unsigned int n_curr_bond = 0;

         for (unsigned int cur_image = 0; cur_image < m_n_images; ++cur_image) // for each image vector
            {
              // make an AABB for the image of this particle
              Scalar r_cut = m_r_cut+m_r_buff;
              vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
              hpmc::detail::AABB aabb = hpmc::detail::AABB(pos_i_image,r_cut);
              hpmc::detail::AABBTree *cur_aabb_tree = &m_aabb_tree;
              // stackless traversal of the tree
              for (unsigned int cur_node_idx = 0; cur_node_idx < cur_aabb_tree->getNumNodes(); ++cur_node_idx)
                  {
                  if (overlap(cur_aabb_tree->getNodeAABB(cur_node_idx), aabb))
                      {
                      if (cur_aabb_tree->isNodeLeaf(cur_node_idx))
                          {
                          for (unsigned int cur_p = 0; cur_p < cur_aabb_tree->getNodeNumParticles(cur_node_idx); ++cur_p)
                              {
                              // neighbor j
                              unsigned int j = cur_aabb_tree->getNodeParticleTag(cur_node_idx, cur_p);

                              if (i!=j)
                                  {
                                  // compute distance
                                  Scalar4 postype_j = h_postype.data[j];
                                  Scalar3 drij = make_scalar3(postype_j.x,postype_j.y,postype_j.z)
                                                 - vec_to_scalar3(pos_i_image);
                                  Scalar dr_sq = dot(drij,drij);

                                  if (dr_sq < r_cutsq)
                                      {

                                      if (n_curr_bond < m_max_bonds)
                                         {

                                         h_nlist.data[i*m_max_bonds + n_curr_bond] = j;

                                         // sort the two tags in this possible bond pair
                                        // const unsigned int tag_a = tag_j>tag_i ? tag_i : tag_j;
                                        // const unsigned int tag_b = tag_j>tag_i ? tag_j : tag_i;

                                         //Scalar3 d = make_scalar3(__int_as_scalar(tag_a),__int_as_scalar(tag_b),dr_sq);

                                        // h_all_possible_bonds.data[group_idx*m_max_bonds + n_curr_bond] = d;
                                         }
                                       else // trigger resize current possible bonds > m_max_bonds
                                         {
                                         m_max_bonds_overflow = std::max(n_curr_bond,m_max_bonds_overflow);
                                         }
                                       ++n_curr_bond;

                                      }
                                  }
                              }
                          }
                      }
                  else
                      {
                      // skip ahead
                      cur_node_idx += cur_aabb_tree->getNodeSkip(cur_node_idx);
                      }
                  } // end stackless search
              } // end loop over images
              h_n_neigh.data[i] = n_curr_bond;
          } // end loop over group 2
   if (m_prof) m_prof->pop();

  /* for( unsigned int i=0; i<size;++i )
   {
     Scalar3 d = h_all_possible_bonds.data[i];
     unsigned int tag_a = __scalar_as_int(d.x);
     unsigned int tag_b = __scalar_as_int(d.y);

     std::cout<< "tree bond  "<< tag_a << " "<< tag_b << " "<< d.z<<std::endl;
   }*/

  }

void DynamicBondUpdater::filterPossibleBonds()
{

  if (m_prof) m_prof->push("filterPossibleBonds");
  //copy data from m_n_list to h_all_possible_bonds
  {
  ArrayHandle<unsigned int> h_nlist(m_n_list, access_location::host, access_mode::read);
  ArrayHandle<unsigned int> h_n_neigh(m_n_neigh, access_location::host, access_mode::read);
  ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
  ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

  ArrayHandle<Scalar3> h_all_possible_bonds(m_all_possible_bonds, access_location::host, access_mode::overwrite);
  unsigned int size = m_group_1->getNumMembers()*m_max_bonds;
  memset((void*) h_all_possible_bonds.data, 0, sizeof(Scalar3)*size);

  const BoxDim& box = m_pdata->getBox();

  // Loop over all particles in group 1
  unsigned int group_size_1 = m_group_1->getNumMembers();
  for (unsigned int group_idx = 0; group_idx < group_size_1; group_idx++)
      {
      unsigned int i = m_group_1->getMemberIndex(group_idx);
      const unsigned int tag_i = h_tag.data[i];
      const Scalar4 postype_i = h_postype.data[i];

      unsigned int n_curr_bond = 0;
      const Scalar r_cutsq = m_r_cut*m_r_cut;

      const unsigned int n_neigh = h_n_neigh.data[i];

      // loop over all neighbors of this particle
      for (unsigned int l=0; l<n_neigh; ++l)
        {
            // get index of neighbor from neigh_list
            const unsigned int j = h_nlist.data[i*m_max_bonds + l];
            Scalar4 postype_j = h_postype.data[j];
            const unsigned int tag_j = h_tag.data[j];

            Scalar3 drij = make_scalar3(postype_j.x,postype_j.y,postype_j.z)
                         - make_scalar3(postype_i.x,postype_i.y,postype_i.z);

           // apply periodic boundary conditions
            drij = box.minImage(drij);

             Scalar dr_sq = dot(drij,drij);

             if (dr_sq < r_cutsq)
                 {
                 if (n_curr_bond < m_max_bonds)
                    {
                    // sort the two tags in this possible bond pair
                    const unsigned int tag_a = tag_j>tag_i ? tag_i : tag_j;
                    const unsigned int tag_b = tag_j>tag_i ? tag_j : tag_i;
                    Scalar3 d = make_scalar3(__int_as_scalar(tag_a),__int_as_scalar(tag_b),dr_sq);
                    h_all_possible_bonds.data[group_idx*m_max_bonds + n_curr_bond] = d;
                    }
                  ++n_curr_bond;
                }

        }
    }
 /*
     for( unsigned int i=0; i<size;++i )
     {
       Scalar3 d = h_all_possible_bonds.data[i];
       unsigned int tag_a = __scalar_as_int(d.x);
       unsigned int tag_b = __scalar_as_int(d.y);

       std::cout<< "tree bond copy "<< tag_a << " "<< tag_b << " "<< d.z<<std::endl;
     }
     */
  }

  //now sort and select down
  {
  m_num_all_possible_bonds = 0;
  ArrayHandle<Scalar3> h_all_possible_bonds(m_all_possible_bonds, access_location::host, access_mode::readwrite);
  const unsigned int size = m_group_1->getNumMembers()*m_max_bonds;

 // remove a possible bond if it already exists. It also removes zeros, e.g.
 // (0,0,0), which fill the unused spots in the array.
 auto last2 = std::remove_if(h_all_possible_bonds.data,
                             h_all_possible_bonds.data + size,
                             [this](Scalar3 i) {return CheckisExistingLegalBond(i); });

 m_num_all_possible_bonds = std::distance(h_all_possible_bonds.data,last2);


  // then sort array by distance between particles in the found possible bond pairs
  // performance is better if remove_if happens before sort
  std::sort(h_all_possible_bonds.data, h_all_possible_bonds.data + m_num_all_possible_bonds, SortBonds);


  // now make sure each possible bond is in the array only once by comparing tags
  auto last = std::unique(h_all_possible_bonds.data, h_all_possible_bonds.data + m_num_all_possible_bonds, CompareBonds);

  m_num_all_possible_bonds = std::distance(h_all_possible_bonds.data,last);
  }


  // at this point, the sub-array: h_all_possible_bonds[0,m_num_all_possible_bonds]
  // should contain only unique entries of possible bonds which are not yet formed.
  if (m_prof) m_prof->pop();
}


/*!
 * (Re-)computes the translation vectors for traversing the BVH tree. At most, there are 27 translation vectors
 * when the simulation box is 3D periodic. In 2D, there are at most 9 translation vectors. In MPI runs, a ghost layer
 * of particles is added from adjacent ranks, so there is no need to perform any translations in this direction.
 * The translation vectors are determined by linear combination of the lattice vectors, and must be recomputed any
 * time that the box resizes.
 */
void DynamicBondUpdater::updateImageVectors()
    {

    const BoxDim& box = m_pdata->getBox();
    uchar3 periodic = box.getPeriodic();
    unsigned char sys3d = (this->m_sysdef->getNDimensions() == 3);

    // now compute the image vectors
    // each dimension increases by one power of 3
    unsigned int n_dim_periodic = (unsigned int)(periodic.x + periodic.y + sys3d*periodic.z);
    m_n_images = 1;
    for (unsigned int dim = 0; dim < n_dim_periodic; ++dim)
        {
        m_n_images *= 3;
        }

    // reallocate memory if necessary
    if (m_n_images > m_image_list.size())
        {
        m_image_list.resize(m_n_images);
        }

    vec3<Scalar> latt_a = vec3<Scalar>(box.getLatticeVector(0));
    vec3<Scalar> latt_b = vec3<Scalar>(box.getLatticeVector(1));
    vec3<Scalar> latt_c = vec3<Scalar>(box.getLatticeVector(2));

    // there is always at least 1 image, which we put as our first thing to look at
    m_image_list[0] = vec3<Scalar>(0.0, 0.0, 0.0);

    // iterate over all other combinations of images, skipping those that are
    unsigned int n_images = 1;
    for (int i=-1; i <= 1 && n_images < m_n_images; ++i)
        {
        for (int j=-1; j <= 1 && n_images < m_n_images; ++j)
            {
            for (int k=-1; k <= 1 && n_images < m_n_images; ++k)
                {
                if (!(i == 0 && j == 0 && k == 0))
                    {
                    // skip any periodic images if we don't have periodicity
                    if (i != 0 && !periodic.x) continue;
                    if (j != 0 && !periodic.y) continue;
                    if (k != 0 && (!sys3d || !periodic.z)) continue;

                    m_image_list[n_images] = Scalar(i) * latt_a + Scalar(j) * latt_b + Scalar(k) * latt_c;
                    ++n_images;
                    }
                }
            }
        }

    forceUpdate();
    }


void DynamicBondUpdater::checkSystemSetup()
{

if (m_bond_type >= m_bond_data -> getNTypes())
  {
  m_exec_conf->msg->error() << "DynamicBondUpdater: bond type id " << m_bond_type  << " is not a valid bond type." << std::endl;
  throw std::runtime_error("Invalid bond type for DynamicBondUpdater");
  }

  if(m_max_bonds_group_1<=0)
  {
  m_exec_conf->msg->warning() << "DynamicBondUpdater: maximum number of bonds for group 1 is <=0. Bonds cannot be formed. " << std::endl;
  }

  if(m_max_bonds_group_2<=0)
  {
  m_exec_conf->msg->warning() << "DynamicBondUpdater: maximum number of bonds for group 2 is <=0. Bonds cannot be formed. " << std::endl;
  }

  if(m_group_1->getNumMembers()<=0)
  {
  m_exec_conf->msg->warning() << "DynamicBondUpdater: group 1 appears to be empty. Bonds cannot be formed. " << std::endl;
  }

  if(m_group_2->getNumMembers()<=0)
  {
  m_exec_conf->msg->warning() << "DynamicBondUpdater: group 2 appears to be empty. Bonds cannot be formed. " << std::endl;
  }

  checkBoxSize();
}

//todo: this function doesn't have a corresponding GPU implementation - what would make sense for this?
void DynamicBondUpdater::makeBonds()
{
  if (m_prof) m_prof->push("makeBonds");

  ArrayHandle<Scalar3> h_all_possible_bonds(m_all_possible_bonds, access_location::host, access_mode::read);
  ArrayHandle<unsigned int> h_n_existing_bonds(m_n_existing_bonds, access_location::host, access_mode::read);

  // we need to count how many bonds are in the h_all_possible_bonds array for a given tag
  // so that we don't end up forming too many bonds in one step. "AddtoExistingBonds" increases the count in
  // h_n_existing_bonds in the for loop below as we go, so no extra bookkeeping should be needed.
  // This also makes it very difficult to port to the gpu.

  ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
  // bool exclusions = m_nlist->getExclusionsSet(); <- doesn't work if we start with a system without any bonds

  //todo: can this for loop be simplified/parallelized?
  for (unsigned int i = 0; i < m_num_all_possible_bonds; i++)
      {
        Scalar3 d = h_all_possible_bonds.data[i];

        unsigned int tag_i = __scalar_as_int(d.x);
        unsigned int tag_j = __scalar_as_int(d.y);
        unsigned int idx_i = h_rtag.data[tag_i];

        // because we save the possible bond pair in an ordered fashion, we actually lost the information in which
        // group tag_i and tag_j is. So we need to look it up.
        bool is_member = m_group_1->isMember(idx_i);
        unsigned int max_bonds_i = is_member ? m_max_bonds_group_1 : m_max_bonds_group_2;
        unsigned int max_bonds_j = is_member ? m_max_bonds_group_2 : m_max_bonds_group_1;

        //todo: put in other external criteria here, e.g. probability of bond formation, max number of bonds possible in one step, etc.
        //todo: randomize which bonds are formed or keep them ordered by their distances?
        if ( max_bonds_i > h_n_existing_bonds.data[tag_i] &&
             max_bonds_j > h_n_existing_bonds.data[tag_j] )
        {
          m_bond_data->addBondedGroup(Bond(m_bond_type,tag_i,tag_j));
          AddtoExistingBonds(tag_i,tag_j);
         if (m_nlist_exclusions_set)  m_nlist->addExclusion(tag_i,tag_j); // this also forces the NeighborList to update
        }
      }

  if (m_prof) m_prof->pop();
}


/*!
 * Check that the largest neighbor search radius is not bigger than twice the shortest box size.
 * Raises an error if this condition is not met. Otherwise, nothing happens.
 */
void DynamicBondUpdater::checkBoxSize()
    {
    const BoxDim& box = m_pdata->getBox();
    const uchar3 periodic = box.getPeriodic();

    // check that rcut fits in the box
    Scalar3 nearest_plane_distance = box.getNearestPlaneDistance();
    Scalar rmax = m_r_cut + m_r_buff;

    if ((periodic.x && nearest_plane_distance.x <= rmax * 2.0) ||
        (periodic.y && nearest_plane_distance.y <= rmax * 2.0) ||
        (m_sysdef->getNDimensions() == 3 && periodic.z && nearest_plane_distance.z <= rmax * 2.0))
        {
        m_exec_conf->msg->error() << "DynamicBondUpdater: Simulation box is too small! Particles would be interacting with themselves." << std::endl;
        throw std::runtime_error("Error in DynamicBondUpdater, Simulation box too small.");
        }
    }


namespace detail
{
/*!
 * \param m Python module to export to
 */
void export_DynamicBondUpdater(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_< DynamicBondUpdater, std::shared_ptr<DynamicBondUpdater> >(m, "DynamicBondUpdater", py::base<Updater>())
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, bool, std::shared_ptr<ParticleGroup>,
             std::shared_ptr<ParticleGroup>, const Scalar,const Scalar, unsigned int, unsigned int, unsigned int>());

        //todo: implement needed getter/setter functions
        //.def_property("inside", &DynamicBondUpdater::getInsideType, &DynamicBondUpdater::setInsideType)
        //.def_property("outside", &DynamicBondUpdater::getOutsideType, &DynamicBondUpdater::setOutsideType)
        //.def_property("lo", &DynamicBondUpdater::getRegionLo, &DynamicBondUpdater::setRegionLo)
        //.def_property("hi", &DynamicBondUpdater::getRegionHi, &DynamicBondUpdater::setRegionHi);
    }
}

} // end namespace azplugins
