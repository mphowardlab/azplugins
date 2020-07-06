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
                         std::shared_ptr<ParticleGroup> group_1,
                         std::shared_ptr<ParticleGroup> group_2,
                         const Scalar r_cut,
                         unsigned int bond_type,
                         unsigned int max_bonds_group_1,
                         unsigned int max_bonds_group_2)
        : Updater(sysdef),  m_group_1(group_1), m_group_2(group_2), m_r_cut(r_cut),
         m_bond_type(bond_type),m_max_bonds_group_1(max_bonds_group_1),m_max_bonds_group_2(max_bonds_group_2),
         m_box_changed(true), m_max_N_changed(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing DynamicBondUpdater" << std::endl;


    m_pdata->getBoxChangeSignal().connect<DynamicBondUpdater, &DynamicBondUpdater::slotBoxChanged>(this);
    m_pdata->getGlobalParticleNumberChangeSignal().connect<DynamicBondUpdater, &DynamicBondUpdater::slotNumParticlesChanged>(this);
  //  m_pdata->getNumTypesChangeSignal().connect<DynamicBondUpdater, &DynamicBondUpdater::slotNumParticlesChanged>(this);

    m_bond_data = m_sysdef->getBondData();

    // allocate initial Memory - grows if necessary
    m_max_bonds = (max_bonds_group_1 < max_bonds_group_2) ? max_bonds_group_2 : max_bonds_group_1;
    m_max_bonds_overflow = 0;
    GPUArray<Scalar3> all_possible_bonds(m_group_2->getNumMembers()*m_max_bonds, m_exec_conf);
    m_all_possible_bonds.swap(all_possible_bonds);

    //todo: reset all aabb componentes if group sizes changes?
    // if groups change this updater might just not work properly - groups don't have a change signal
    m_aabbs.resize(m_group_1->getNumMembers());

    checkSystemSetup();

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
void DynamicBondUpdater::update(unsigned int timestep)
{
    // update properties that depend on the box
    if (m_box_changed)
        {
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
    do
        {
        calculatePossibleBonds();
        overflowed = m_max_bonds < m_max_bonds_overflow;
        // if we overflowed, need to reallocate memory and re-calculate
        if (overflowed)
            {
            resizePossibleBondlists();
            }
        } while (overflowed);

    filterPossibleBonds();
    // this function is not easily implemented on the GPU, uses addBondedGroup()
    makeBonds();


}

//todo: should go into helper class?
bool SortBonds(Scalar3 i, Scalar3 j)
  {
      const Scalar r_sq_1 = i.z;
      const Scalar r_sq_2 = j.z;
      return r_sq_1 < r_sq_2;
  }

//todo: should go into helper class? - also, faster way without branching?
  bool CompareBonds(Scalar3 i, Scalar3 j)
  {

      const unsigned int tag_11 = __scalar_as_int(i.x);
      const unsigned int tag_12 = __scalar_as_int(i.y);
      const unsigned int tag_21 = __scalar_as_int(j.x);
      const unsigned int tag_22 = __scalar_as_int(j.y);

      if ((tag_11==tag_21 && tag_12==tag_22) ||   // (i,j)==(i,j)
          (tag_11==tag_22 && tag_12==tag_21))     // (i,j)==(j,i)
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
//  ArrayHandle<typeval_t> h_typeval(m_bond_data->getTypeValArray(), access_location::host, access_mode::read);

  // for each of the bonds
  const unsigned int size = (unsigned int)m_bond_data->getN();
  for (unsigned int i = 0; i < size; i++)
  {
      // lookup the tag of each of the particles participating in the bond
      const typename BondData::members_t& bond = h_bonds.data[i];
      unsigned int tag1 = bond.tag[0];
      unsigned int tag2 = bond.tag[1];
    //  unsigned int type = h_typeval.data[i].type;

      //only keep track of the bond type we are forming - does this make sense?
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

  // don't add a bond twice - should not happen anyway
  if (isExistingBond(tag1, tag2)) return;

  bool overflowed = false;

  ArrayHandle<unsigned int> h_n_existing_bonds(m_n_existing_bonds, access_location::host, access_mode::readwrite);

  // resize the list if necessary
  if (h_n_existing_bonds.data[tag1] == m_existing_bonds_list_indexer.getH())
      overflowed = true;

  if (h_n_existing_bonds.data[tag2] == m_existing_bonds_list_indexer.getH())
      overflowed = true;


  if (overflowed) resizeExistingBondList();

  ArrayHandle<unsigned int> h_existing_bonds_list(m_existing_bonds_list, access_location::host, access_mode::readwrite);

  // add tag2 to tag1's exclusion list
  unsigned int pos1 = h_n_existing_bonds.data[tag1];
  assert(pos1 < m_existing_bonds_list_indexer.getH());
  h_existing_bonds_list.data[m_existing_bonds_list_indexer(tag1,pos1)] = tag2;
  h_n_existing_bonds.data[tag1]++;

  // add tag1 to tag2's exclusion list
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
    }

//todo: should the list be grown more than 1 at a time for efficiency?
void DynamicBondUpdater::resizePossibleBondlists()
    {
      m_max_bonds=m_max_bonds_overflow;
      m_max_bonds_overflow=0;
      unsigned int size = m_group_2->getNumMembers()*m_max_bonds;
      m_all_possible_bonds.resize(size);

      m_exec_conf->msg->notice(6) << "DynamicBondUpdater: (Re-)size possible bond list, new size " << m_max_bonds << " bonds per particle " << std::endl;
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

    calculateExistingBonds();
  }

void DynamicBondUpdater::calculatePossibleBonds()
  {
    //todo: is it worth it to seperate the tree building out and check if update is necessary similar to neighbor list?
    // make tree for group 1
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<hpmc::detail::AABB> h_aabbs(m_aabbs, access_location::host, access_mode::readwrite);

    unsigned int group_size_1 = m_group_1->getNumMembers();
    for (unsigned int group_idx = 0; group_idx < group_size_1; group_idx++)
        {
        unsigned int i = m_group_1->getMemberIndex(group_idx);
        // make a point particle AABB
        vec3<Scalar> my_pos(h_postype.data[i]);
        h_aabbs.data[group_idx] = hpmc::detail::AABB(my_pos,i);
        }

    m_aabb_tree.buildTree(&(h_aabbs.data[0]) , group_size_1);

     // reset content of possible bond list
     ArrayHandle<Scalar3> h_all_possible_bonds(m_all_possible_bonds, access_location::host, access_mode::overwrite);
     const unsigned int size = m_group_2->getNumMembers()*m_max_bonds;
     memset((void*)h_all_possible_bonds.data, 0, sizeof(Scalar3)*size);

     // traverse the tree
     // Loop over all particles in group 2
     unsigned int group_size_2 = m_group_2->getNumMembers();
     for (unsigned int group_idx = 0; group_idx < group_size_2; group_idx++)
         {
         unsigned int i = m_group_2->getMemberIndex(group_idx);
         const unsigned int tag_i = h_tag.data[i];
         const Scalar4 postype_i = h_postype.data[i];
         const vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

         unsigned int n_curr_bond = 0;

        for (unsigned int cur_image = 0; cur_image < m_n_images; ++cur_image) // for each image vector
           {
             // make an AABB for the image of this particle
             vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
             hpmc::detail::AABB aabb = hpmc::detail::AABB(pos_i_image, m_r_cut);
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

                             // skip self-interaction always
                             bool excluded = (i == j);

                             if (!excluded)
                                 {
                                 // compute distance
                                 Scalar4 postype_j = h_postype.data[j];
                                 Scalar3 drij = make_scalar3(postype_j.x,postype_j.y,postype_j.z)
                                                - vec_to_scalar3(pos_i_image);
                                 Scalar dr_sq = dot(drij,drij);
                                 const Scalar r_cutsq = m_r_cut*m_r_cut;
                                 if (dr_sq <= r_cutsq)
                                     {
                                     if (n_curr_bond < m_max_bonds)
                                        {
                                        const unsigned int tag_j = h_tag.data[j];
                                        Scalar3 d ;
                                        d = make_scalar3(__int_as_scalar(tag_j),__int_as_scalar(tag_i),dr_sq);
                                        h_all_possible_bonds.data[group_idx + n_curr_bond] = d;
                                        }
                                      else // trigger resize current possible bonds > m_max_bonds
                                        {
                                        m_max_bonds_overflow = n_curr_bond;
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
         } // end loop over group 2

 }

void DynamicBondUpdater::filterPossibleBonds()
{

  ArrayHandle<Scalar3> h_all_possible_bonds(m_all_possible_bonds, access_location::host, access_mode::overwrite);
  const unsigned int size = m_group_2->getNumMembers()*m_max_bonds;

// first sort whole array by distance between particles in that particular possible bond
std::sort(h_all_possible_bonds.data, h_all_possible_bonds.data + size, SortBonds);

// now make sure each possible bond is in the array only once by comparing tags
auto last = std::unique(h_all_possible_bonds.data, h_all_possible_bonds.data + size, CompareBonds);
m_num_all_possible_bonds = std::distance(h_all_possible_bonds.data,last);

// then remove a possible bond if it already exists. It also removes zeros, e.g.
// (0,0,0), which fill the unused spots in the array.
auto last2 = std::remove_if(h_all_possible_bonds.data,
                            h_all_possible_bonds.data + m_num_all_possible_bonds,
                           [this](Scalar3 i) {return CheckisExistingLegalBond(i); });

m_num_all_possible_bonds = std::distance(h_all_possible_bonds.data,last2);

// at this point, the sub-array: h_all_possible_bonds[0,m_num_all_possible_bonds]
// should contain only unique entries of possible bonds which are not yet formed.
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
    }


void DynamicBondUpdater::checkSystemSetup()
{

if (m_bond_type >= m_bond_data -> getNTypes())
  {
  m_exec_conf->msg->error() << "DynamicBondUpdater: bond type id " << m_bond_type
                            << " is not a valid bond type." << std::endl;
  throw std::runtime_error("Invalid bond type for DynamicBondUpdater");
  }

}

void DynamicBondUpdater::makeBonds()
{

  // we need to count how many bonds are in the h_all_possible_bonds array for a given tag
  // so that we don't end up forming too many bonds in one step
  ArrayHandle<Scalar3> h_all_possible_bonds(m_all_possible_bonds, access_location::host, access_mode::read);
  ArrayHandle<unsigned int> h_n_existing_bonds(m_n_existing_bonds, access_location::host, access_mode::read);
  ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

  GPUArray<unsigned int> total_counts(m_pdata->getRTags().size(), m_exec_conf);
  ArrayHandle<unsigned int> h_total_counts(total_counts, access_location::host, access_mode::readwrite);
  memset((void*)h_total_counts.data,0,sizeof(unsigned int)*m_pdata->getRTags().size());

  // Loop over all possible bonds and count how many times a tag is in h_all_possible_bonds array
  // save how many bonds to form (max bonds - existing bonds) in h_total_counts
  for (unsigned int i = 0; i < m_num_all_possible_bonds; i++)
      {
        Scalar3 d = h_all_possible_bonds.data[i];
        const  int tag_i = __scalar_as_int(d.x);
        const  int tag_j = __scalar_as_int(d.y);
        h_total_counts.data[tag_i]=m_max_bonds_group_1 - h_n_existing_bonds.data[tag_i];
        h_total_counts.data[tag_j]=m_max_bonds_group_2 - h_n_existing_bonds.data[tag_j];
      }

  GPUArray<unsigned int> current_counts(m_pdata->getRTags().size(), m_exec_conf);
  ArrayHandle<unsigned int> h_current_counts(current_counts, access_location::host, access_mode::readwrite);
  memset((void*)h_current_counts.data,0,sizeof(unsigned int)*m_pdata->getRTags().size());

  //todo: can this for loop be simplified/paralleized?
  bool added_bonds = false;
  for (unsigned int i = 0; i < m_num_all_possible_bonds; i++)
      {
        Scalar3 d = h_all_possible_bonds.data[i];
        unsigned int tag_i = __scalar_as_int(d.x);
        unsigned int tag_j = __scalar_as_int(d.y);

        //todo: put in other external criteria here, e.g. probability of bond formation etc
        if (h_current_counts.data[tag_i]<h_total_counts.data[tag_i] &&
        h_current_counts.data[tag_j]< h_total_counts.data[tag_j] )
        {
          m_bond_data->addBondedGroup(Bond(m_bond_type,tag_i,tag_j));
          AddtoExistingBonds(tag_i,tag_j);
          h_current_counts.data[tag_i]++;
          h_current_counts.data[tag_j]++;
          added_bonds = true;
        }
      }

 if (added_bonds) m_pdata->notifyParticleSort();

 //todo: how to add this? m_nlist as parameter? also needs to know if exclusions are set in nlist
 //notify neighbor lists
//  if (m_exclude_from_nlist)
//    m_nlist->addExclusion(p_from_idx,p_to_idx);

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
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>,
             std::shared_ptr<ParticleGroup>, const Scalar, unsigned int, unsigned int, unsigned int>());

        //.def_property("inside", &DynamicBondUpdater::getInsideType, &DynamicBondUpdater::setInsideType)
        //.def_property("outside", &DynamicBondUpdater::getOutsideType, &DynamicBondUpdater::setOutsideType)
        //.def_property("lo", &DynamicBondUpdater::getRegionLo, &DynamicBondUpdater::setRegionLo)
        //.def_property("hi", &DynamicBondUpdater::getRegionHi, &DynamicBondUpdater::setRegionHi);
    }
}

} // end namespace azplugins
