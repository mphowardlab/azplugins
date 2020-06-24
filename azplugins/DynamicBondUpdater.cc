// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file DynamicBondUpdater.cc
 * \brief Definition of DynamicBondUpdater
 */

#include "DynamicBondUpdater.h"


namespace azplugins
{


/*!
 * \param sysdef System definition
 * \param inside_type Type id of particles inside region
 * \param outside_type Type id of particles outside region
 * \param z_lo Lower bound of region in z
 * \param z_hi Upper bound of region in z
 */
DynamicBondUpdater::DynamicBondUpdater(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<NeighborList> nlist,
                         std::shared_ptr<ParticleGroup> group_1,
                         std::shared_ptr<ParticleGroup> group_2,
                         const Scalar r_cutsq,
                         unsigned int bond_type,
                         unsigned int bond_reservoir_type,
                         unsigned int max_bonds_group_1,
                         unsigned int max_bonds_group_2)
        : Updater(sysdef), m_nlist(nlist), m_group_1(group_1), m_group_2(group_2), m_r_cutsq(r_cutsq),
         m_bond_type(bond_type),m_bond_reservoir_type(bond_reservoir_type),m_max_bonds_group_1(max_bonds_group_1),m_max_bonds_group_2(max_bonds_group_2)
    {
    m_exec_conf->msg->notice(5) << "Constructing DynamicBondUpdater" << std::endl;

    assert(m_nlist);

    m_bond_data = m_sysdef->getBondData();
    // allocate memory for the number of current bonds array
    GPUArray<unsigned int> counts((int)m_pdata->getN(), m_exec_conf);
    m_curr_num_bonds.swap(counts);

    m_max_bonds=20;
    // allocate a max size for all possible pairs - is there a better way to do this?
    // could model the neighbor list m_conditions
    const unsigned int size = m_group_2->getNumMembers()*m_max_bonds;
    GPUArray<Scalar3> all_possible_bonds(size, m_exec_conf);
    m_all_possible_bonds.swap(all_possible_bonds);

    m_aabbs.resize(m_group_1->getNumMembers());

    checkSystemSetup();
    updateImageVectors();
    calculateCurrentBonds();

    }

DynamicBondUpdater::~DynamicBondUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying DynamicBondUpdater" << std::endl;

    }

bool SortBonds(Scalar3 i, Scalar3 j)
  {
      const Scalar r_sq_1 = i.z;
      const Scalar r_sq_2 = j.z;
      return r_sq_1 < r_sq_2;
  }

  // Function for binary_predicate
  bool CompareBonds(Scalar3 i, Scalar3 j)
  {

      const unsigned int tag_11 = __scalar_as_int(i.x);
      const unsigned int tag_12 = __scalar_as_int(i.y);
      const unsigned int tag_21 = __scalar_as_int(j.x);
      const unsigned int tag_22 = __scalar_as_int(j.y);

      if (tag_11==tag_21 && tag_12==tag_22)
      {
        return true;
      }else{
        return false;
      }
  }



void DynamicBondUpdater::calculateCurrentBonds()
{

  // std::cout<< " in DynamicBondUpdater::calculateCurrentBonds() "<<std::endl;
    //make tree for group 1
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<hpmc::detail::AABB> h_aabbs(m_aabbs, access_location::host, access_mode::readwrite);

    const BoxDim& box = m_pdata->getBox();

    unsigned int group_size_1 = m_group_1->getNumMembers();
    for (unsigned int group_idx = 0; group_idx < group_size_1; group_idx++)
        {
        unsigned int i = m_group_1->getMemberIndex(group_idx);

        // make a point particle AABB
        vec3<Scalar> my_pos(h_postype.data[i]);
      //  std::cout<< "particle "<< i << " in group 1 "<< group_idx<<std::endl;
        h_aabbs.data[group_idx] = hpmc::detail::AABB(my_pos,i);
        }
    //    std::cout<< " in DynamicBondUpdater::calculateCurrentBonds() after group loop "<<std::endl;
    // call the tree build routine
    m_aabb_tree.buildTree(&(h_aabbs.data[0]) , group_size_1);
  //  std::cout<< " in DynamicBondUpdater::calculateCurrentBonds() after buildtree"<<std::endl;

     //traverse tree

     // acquire particle data

     ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
     //ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);

     //ArrayHandle<Scalar> h_r_cut(m_r_cut, access_location::host, access_mode::read);

     // reset possible bond list
     ArrayHandle<Scalar3> h_all_possible_bonds(m_all_possible_bonds, access_location::host, access_mode::overwrite);
     const unsigned int size = m_group_2->getNumMembers()*m_max_bonds;
     memset((void*)h_all_possible_bonds.data, 0, sizeof(Scalar3)*size);

     Scalar r_cut = 3.0;
     Scalar m_r_buff = 0.4;
     bool m_filter_body = false;
     // neighborlist data
    // ArrayHandle<unsigned int> h_head_list(m_head_list, access_location::host, access_mode::read);
    // ArrayHandle<unsigned int> h_Nmax(m_Nmax, access_location::host, access_mode::read);
    // ArrayHandle<unsigned int> h_conditions(m_conditions, access_location::host, access_mode::readwrite);
  //   ArrayHandle<unsigned int> h_nlist(m_nlist, access_location::host, access_mode::overwrite);
  //   ArrayHandle<unsigned int> h_n_neigh(m_n_neigh, access_location::host, access_mode::overwrite);

     // Loop over all particles in group 2
     //std::cout<< " inDynamicBondUpdater::calculateCurrentBonds() before particle loop"<<std::endl;
     unsigned int group_size_2 = m_group_2->getNumMembers();
     for (unsigned int group_idx = 0; group_idx < group_size_2; group_idx++)
         {
         unsigned int i = m_group_2->getMemberIndex(group_idx);
         const unsigned int tag_i = h_tag.data[i];

         const Scalar4 postype_i = h_postype.data[i];
         const vec3<Scalar> pos_i = vec3<Scalar>(postype_i);
         const unsigned int type_i = __scalar_as_int(postype_i.w);
         const unsigned int body_i = h_body.data[i];

         unsigned int n_curr_bond = 0;

        for (unsigned int cur_image = 0; cur_image < m_n_images; ++cur_image) // for each image vector
           {
             // make an AABB for the image of this particle
             vec3<Scalar> pos_i_image = pos_i + m_image_list[cur_image];
             hpmc::detail::AABB aabb = hpmc::detail::AABB(pos_i_image, r_cut);
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
                                 Scalar r_cutsq = r_cut*r_cut;
                                 if (dr_sq <= r_cutsq)
                                     {
                                     if (n_curr_bond < m_max_bonds)
                                          {
                                          const unsigned int tag_j = h_tag.data[j];
                                          Scalar3 d ;
                                          if( tag_i < tag_j )
                                          {
                                            d = make_scalar3(__int_as_scalar(tag_i),__int_as_scalar(tag_j),dr_sq);
                                          }else{
                                            d = make_scalar3(__int_as_scalar(tag_j),__int_as_scalar(tag_i),dr_sq);
                                          }
                                          h_all_possible_bonds.data[group_idx + n_curr_bond] = d;
                                          ++n_curr_bond;
                                          }

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
         //} // end loop over pair types
         } // end loop over group 2
//thrust::copy_if(input.begin(), input.end(), output.begin(), is_non_zero<int>());
//Now we call the sort function

//std::copy_if(h_all_possible_bonds.data,is_set<Scalar3>())
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


    if (m_bond_reservoir_type >= m_bond_data -> getNTypes())
        {
        m_exec_conf->msg->error() << "DynamicBondUpdater: bond type id " << m_bond_reservoir_type
                                  << " is not a valid bond type." << std::endl;
        throw std::runtime_error("Invalid bond type for DynamicBondUpdater");
        }

    //ArrayHandle<unsigned int> h_curr_num_bonds(m_curr_num_bonds, access_location::host, access_mode::readwrite);

    if (m_reservoir_size==0)
        {
        m_exec_conf->msg->error() << "DynamicBondUpdater: Bond reservoir size is zero." << std::endl;
        throw std::runtime_error("DynamicBondUpdater: Bond reservoir size must be larger than zero.");
        }

}

/*!
 * \param timestep Timestep update is called
 */
void DynamicBondUpdater::update(unsigned int timestep)
    {

    calculateCurrentBonds();
    findPotentialBondPairs(timestep);

  //  formBondPairs(timestep);

    }

/*!
 * \param timestep Timestep update is called
 */
void DynamicBondUpdater::findPotentialBondPairs(unsigned int timestep)
    {

    ArrayHandle<Scalar3> h_all_possible_bonds(m_all_possible_bonds, access_location::host, access_mode::readwrite);
    const unsigned int size = m_group_2->getNumMembers()*m_max_bonds;

    std::sort(h_all_possible_bonds.data, h_all_possible_bonds.data + size,SortBonds);
    Scalar3 *last = std::unique(h_all_possible_bonds.data, h_all_possible_bonds.data + size,CompareBonds);
    std::cout<<" last "<< last->x<<" "<< last->y << " "<<last->z <<std::endl;
    for (unsigned int i = 0; i < size; i++)
        {
          Scalar3 d = h_all_possible_bonds.data[i];
          unsigned int tag_i = __scalar_as_int(d.x);
          unsigned int tag_j = __scalar_as_int(d.y);
          Scalar r_sq = d.z;
          std::cout<< "possible bond "<< i << " tag_i "<< tag_i << " tag_j "<< tag_j<<" dist_sq "<< r_sq<< std::endl;

        }
    /*
    // start by updating the neighborlist
    //m_nlist->addExclusionsFromBonds();
    m_nlist->compute(timestep);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle< unsigned int > h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // neighbour list information
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);
    // box
    const BoxDim& box = m_pdata->getGlobalBox();

    ArrayHandle<unsigned int> h_curr_num_bonds(m_curr_num_bonds, access_location::host, access_mode::read);

    // temp vector storage of possible bonds found
    std::vector<Scalar2> possible_bonds;

    // for each particle in group_1 - parallelize the outer loop on the GPU?
    for (unsigned int i = 0; i < m_group_1->getNumMembers(); ++i)
        {
        // get particle index
        const unsigned int idx_i = m_group_1->getMemberIndex(i);
        const unsigned int tag_i = h_tag.data[idx_i];
        //const unsigned int idx_i = h_group_1.data[i];

        // loop over all of the neighbors of this particle
        const unsigned int myHead = h_head_list.data[idx_i];
        const unsigned int size = (unsigned int)h_n_neigh.data[idx_i];
        unsigned int num_possible_bonds_i = 0;

        // this way of finding neighbours introduces some artifacts if the particle index and spatial positions are
        // correlated, because the neighbor list returns neighbours ordered by index, so the particles with lower index
        // get bonded first if the number of possible bonds exceedst the bond limit. if there is spatial correlation
        // this could lead to artifacts in the spatial configuration as well. Is it worth it to shuffle the order?
        for (unsigned int k = 0; k < size; k++)
            {
            // access the index of this neighbor
            const unsigned int idx_j = h_nlist.data[myHead + k];
            const unsigned int tag_j = h_tag.data[idx_j];

            bool is_in_group_2 = m_group_2->isMember(idx_j); // needs to be replaced with something else on the GPU

            const unsigned int current_bonds_on_j = h_curr_num_bonds.data[tag_j];
            const unsigned int current_bonds_on_i = h_curr_num_bonds.data[tag_i];

            // check that this bond doesn't already exists, second particle is in second group, and max number of bonds is not reached for both
            if (is_in_group_2
                && m_all_existing_bonds.count({tag_i, tag_j}) ==0
                && m_all_existing_bonds.count({tag_j, tag_i}) ==0
                && current_bonds_on_j < m_max_bonds_group_2
                && num_possible_bonds_i < m_max_bonds_group_1-current_bonds_on_i )
                {
                //  caclulate distance squared
                const Scalar3 pi = make_scalar3(h_pos.data[idx_i].x, h_pos.data[idx_i].y, h_pos.data[idx_i].z);
                const Scalar3 pj = make_scalar3(h_pos.data[idx_j].x, h_pos.data[idx_j].y, h_pos.data[idx_j].z);
                Scalar3 dx = pi - pj;
                dx = box.minImage(dx);
                const Scalar rsq = dot(dx, dx);

                if (rsq < m_r_cutsq)
                    {
                    possible_bonds.push_back(make_scalar2(tag_i,tag_j));
                    //std::cout<< "added bond to possible list "<< idx_i << " "<< idx_j << "tag "<< tag_i << " "<< tag_j<< std::endl;
                    ++num_possible_bonds_i;
                    }

                }
            }


        }

    // reset possible bond list
    ArrayHandle<Scalar3> h_possible_bonds(m_all_possible_bonds, access_location::host, access_mode::overwrite);
    const unsigned int size = m_group_1->getNumMembers()*m_max_bonds_group_1;
    memset((void*)h_possible_bonds.data,-1.0,sizeof(Scalar2)*size);

    // Before we copy the possible_bonds vector content to the h_possible_bonds array,  we need count number of bonds
    // formed towards particles in group_2 (second entry in possible bonds) because there could be too many.
    // The group_1 bonds should be okay because we are able to check in the for loop above.

     // a temp map which holds count of each encountered particle tag in group_2
    std::unordered_map<int, size_t> count_group_2_possible_bonds;

    // iterate over all possible bonds and use the unordered_map to count occurences, if occurences is larger than max_bonds_group_2
    // then don't copy that entry into the h_possible_bonds Array
    unsigned int current = 0;
    for (auto i = possible_bonds.begin(); i != possible_bonds.end(); ++i)
        {
        unsigned int tag_j = i->y;
        ++count_group_2_possible_bonds[tag_j];
        const unsigned int current_bonds_on_j = h_curr_num_bonds.data[tag_j];
        if(count_group_2_possible_bonds[tag_j] + current_bonds_on_j < m_max_bonds_group_2)
            {
          //  h_possible_bonds.data[current]=make_scalar2(i->x,i->y);
            ++current;
            }
        }

     m_curr_bonds_to_form = current;
     */
    }


void DynamicBondUpdater::formBondPairs(unsigned int timestep)
    {


    ArrayHandle<Scalar3> h_possible_bonds(m_all_possible_bonds, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_curr_num_bonds(m_curr_num_bonds, access_location::host, access_mode::readwrite);

    ArrayHandle<typename BondData::members_t> h_bonds(m_bond_data->getMembersArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<typeval_t> h_typeval(m_bond_data->getTypeValArray(), access_location::host, access_mode::readwrite);
//    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // only do something if there are bonds to form and there are blank bonds left
    if (m_curr_bonds_to_form>0 && m_reservoir_size>0)
        {

        const unsigned int size = (unsigned int)m_bond_data->getN();
        unsigned int current = 0;
        for (unsigned int i = 0; i < size; i++)
            {

            unsigned int type = h_typeval.data[i].type;

            if (type == m_bond_reservoir_type && current < m_curr_bonds_to_form)
                {
                h_typeval.data[i].type = m_bond_type;
                unsigned int tag_i = h_possible_bonds.data[current].x;
                unsigned int tag_j = h_possible_bonds.data[current].y;

                h_bonds.data[i].tag[0] =  tag_i;
                h_bonds.data[i].tag[1] =  tag_j;

                //add new bond to the book keeping arrays and the map
                ++h_curr_num_bonds.data[tag_i];
                ++h_curr_num_bonds.data[tag_j];
                m_all_existing_bonds[{tag_i,tag_j}]=1;
                m_all_existing_bonds[{tag_i,tag_j}]=1;

                ++current;
                --m_reservoir_size;
                }
            }
        }
        m_curr_bonds_to_form=0;

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
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, std::shared_ptr<ParticleGroup>,
             std::shared_ptr<ParticleGroup>, const Scalar, unsigned int, unsigned int, unsigned int, unsigned int>());

        //.def_property("inside", &DynamicBondUpdater::getInsideType, &DynamicBondUpdater::setInsideType)
        //.def_property("outside", &DynamicBondUpdater::getOutsideType, &DynamicBondUpdater::setOutsideType)
        //.def_property("lo", &DynamicBondUpdater::getRegionLo, &DynamicBondUpdater::setRegionLo)
        //.def_property("hi", &DynamicBondUpdater::getRegionHi, &DynamicBondUpdater::setRegionHi);
    }
}

} // end namespace azplugins
