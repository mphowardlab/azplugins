// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file DynamicBondUpdaterGPU.cc
 * \brief Definition of DynamicBondUpdaterGPU
 */

#include "DynamicBondUpdaterGPU.h"
#include "DynamicBondUpdaterGPU.cuh"

namespace azplugins
{

 DynamicBondUpdaterGPU::DynamicBondUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                                              std::shared_ptr<NeighborList> nlist,
                                              std::shared_ptr<ParticleGroup> group_1,
                                              std::shared_ptr<ParticleGroup> group_2,
                                              const Scalar r_cut,
                                              const Scalar r_buff,
                                              unsigned int bond_type,
                                              unsigned int max_bonds_group_1,
                                              unsigned int max_bonds_group_2)
        : DynamicBondUpdater(sysdef, nlist, group_1, group_2, r_cut, r_buff,bond_type, max_bonds_group_1, max_bonds_group_2),
        m_num_nonzero_bonds(m_exec_conf),m_max_bonds_overflow_flag(m_exec_conf),
        m_lbvh_errors(m_exec_conf),m_lbvh_2(m_exec_conf),m_traverser(m_exec_conf)
    {
    m_sorted_index_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "dynamic_bond_updater_sorted_index", m_exec_conf));
    m_copy_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "dynamic_bond_updater_tree_copy", m_exec_conf));
    m_copy_nlist_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "dynamic_bond_updater_nlist_copy", m_exec_conf));
    m_tuner_filter_bonds.reset(new Autotuner(32, 1024, 32, 5, 100000, "dynamic_bond_updater_filter_bonds", m_exec_conf));
    m_count_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "dynamic_bond_updater_count", m_exec_conf));

    // allocate initial Memory - grows if necessary
    GPUArray<Scalar3> all_possible_bonds(m_group_1->getNumMembers()*m_max_bonds, m_exec_conf);
    m_all_possible_bonds.swap(all_possible_bonds);

    checkSystemSetup();
    }

DynamicBondUpdaterGPU::~DynamicBondUpdaterGPU()
    {
      m_exec_conf->msg->notice(5) << "Destroying DynamicBondUpdaterGPU" << std::endl;
    }


void DynamicBondUpdaterGPU::buildTree()
    {
    if (m_prof) m_prof->push("buildTree");

     // setup the sorted index, we already know the indexes of the particles in
     // the two groups, we can simply copy them into the m_sorted_indexes array.
     ArrayHandle<unsigned int> d_sorted_indexes(m_sorted_indexes, access_location::device, access_mode::overwrite);
     ArrayHandle<unsigned int> d_index_group_1(m_group_1->getIndexArray(), access_location::device, access_mode::read);
     ArrayHandle<unsigned int> d_index_group_2(m_group_2->getIndexArray(), access_location::device, access_mode::read);

     m_sorted_index_tuner->begin();
     azplugins::gpu::make_sorted_index_array(d_sorted_indexes.data,
                                             d_index_group_1.data,
                                             d_index_group_2.data,
                                             m_group_1->getNumMembers(),
                                             m_group_2->getNumMembers(),
                                             m_count_tuner->getParam());
       if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
       m_sorted_index_tuner->end();

    // build a lbvh for group_2
    {
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_sorted_indexes(m_sorted_indexes, access_location::device, access_mode::read);

    const BoxDim lbvh_box = getLBVHBox();

    // this tree is traversed in traverseTree()
    m_lbvh_2.build(azplugins::gpu::PointMapInsertOp(d_pos.data,
                                d_sorted_indexes.data + m_group_1->getNumMembers(),
                                m_group_2->getNumMembers()),
                                lbvh_box.getLo(),
                                lbvh_box.getHi());

   }
   if (m_prof) m_prof->pop();
    }

void DynamicBondUpdaterGPU::traverseTree()
    {
   if (m_prof) m_prof->push("traverseTree");
    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_n_neigh(m_n_neigh, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_sorted_indexes(m_sorted_indexes, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

    // clear the neighbor counts
    cudaMemset(d_n_neigh.data, 0, sizeof(unsigned int)*m_pdata->getMaxN());
    cudaMemset( d_nlist.data,0, sizeof(unsigned int)*m_max_bonds*m_pdata->getMaxN());

    const BoxDim& box = m_pdata->getBox();

    // neighbor list write op
    azplugins::gpu::NeighborListOp nlist_op(d_nlist.data, d_n_neigh.data, m_max_bonds_overflow_flag.getDeviceFlags(), m_max_bonds);

    neighbor::MapTransformOp map(d_sorted_indexes.data + m_group_1->getNumMembers());
    m_traverser.setup(map, m_lbvh_2);

    // todo: use sorted indexes as traverse order? Is that ok?
    azplugins::gpu::ParticleQueryOp query_op(d_pos.data,
                                               d_sorted_indexes.data + 0,
                                               m_group_1->getNumMembers(),
                                               m_pdata->getMaxN(),
                                               m_r_cut+m_r_buff,
                                               box);

     m_traverser.traverse(nlist_op, query_op, map, m_lbvh_2, m_image_list);

     m_max_bonds_overflow =  m_max_bonds_overflow_flag.readFlags();

     // if we didn't overflow copy information from nlist to all_possible_bonds array, do distance checking
     // if it did overflow traverse tree again first to put all neighbor information into nlist and n_neigh
     if( m_max_bonds_overflow <= m_max_bonds)
     {

       ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::read);
       ArrayHandle<unsigned int> d_n_neigh(m_n_neigh, access_location::device, access_mode::read);
       ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
       ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
       ArrayHandle<unsigned int> d_sorted_indexes(m_sorted_indexes, access_location::device, access_mode::read);

       ArrayHandle<Scalar3> d_all_possible_bonds(m_all_possible_bonds, access_location::device, access_mode::overwrite);
       unsigned int size = m_group_1->getNumMembers()*m_max_bonds;
       cudaMemset((void*) d_all_possible_bonds.data, 0, sizeof(Scalar3)*size);


       const BoxDim& box = m_pdata->getBox();
       m_copy_nlist_tuner->begin();
       azplugins::gpu::nlist_copy_nlist_possible_bonds(d_all_possible_bonds.data,
                                 d_pos.data,
                                 d_tag.data,
                                 d_sorted_indexes.data,
                                 d_n_neigh.data,
                                 d_nlist.data,
                                 box,
                                 m_max_bonds,
                                 m_r_cut,
                                 m_group_1->getNumMembers(),
                                 m_copy_tuner->getParam());
       if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
       m_copy_nlist_tuner->end();

     }
    if (m_prof) m_prof->pop();
    }


void DynamicBondUpdaterGPU::resizePossibleBondlists()
    {
      // round up to nearest multiple of 4
      m_max_bonds_overflow = (m_max_bonds_overflow > 4) ? (m_max_bonds_overflow + 3) & ~3 : 4;
      m_max_bonds = m_max_bonds_overflow;
      m_max_bonds_overflow = 0;
      unsigned int size = m_group_1->getNumMembers()*m_max_bonds;
      m_all_possible_bonds.resize(size);
      m_num_all_possible_bonds=0;

      GlobalArray<unsigned int> nlist(m_max_bonds*m_pdata->getMaxN(), m_exec_conf);
      m_nlist.swap(nlist);

      m_exec_conf->msg->notice(6) << "DynamicBondUpdaterGPU: (Re-)size possible bond list, new size " << m_max_bonds << " bonds per particle " << std::endl;

    }


void DynamicBondUpdaterGPU::allocateParticleArrays()
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

    GPUArray<unsigned int> sorted_indexes(m_pdata->getMaxN(), m_exec_conf);
    m_sorted_indexes.swap(sorted_indexes);

    // allocate m_last_pos
    GlobalArray<Scalar4> last_pos(m_pdata->getMaxN(), m_exec_conf);
    m_last_pos.swap(last_pos);

    // allocate the number of neighbors (per particle)
    GlobalArray<unsigned int> n_neigh(m_pdata->getMaxN(), m_exec_conf);
    m_n_neigh.swap(n_neigh);
    ArrayHandle<unsigned int> h_n_neigh(m_n_neigh, access_location::host, access_mode::overwrite);
    memset(h_n_neigh.data, 0, sizeof(unsigned int)*m_pdata->getMaxN());

    // default allocation of m_max_bonds neighbors per particle for the neighborlist
    GlobalArray<unsigned int> nlist(m_max_bonds*m_pdata->getMaxN(), m_exec_conf);
    m_nlist.swap(nlist);

    calculateExistingBonds();

  }



void DynamicBondUpdaterGPU::filterPossibleBonds()
  {
  if (m_prof) m_prof->push("filterPossibleBonds1");
  // todo: figure out in which order the thrust calls are the fastest.
  // is using build in thrust functions the best solution?
  // suspect: sort - remove zeros - unique - filter (which introduces zeros) - remove zeros ?
  m_num_all_possible_bonds = 0;
  const unsigned int size = m_group_1->getNumMembers()*m_max_bonds;

  // sort and remove all existing zeros
  ArrayHandle<unsigned int> d_n_existing_bonds(m_n_existing_bonds, access_location::device, access_mode::read);
  ArrayHandle<unsigned int> d_existing_bonds_list(m_existing_bonds_list, access_location::device, access_mode::read);
  ArrayHandle<Scalar3> d_all_possible_bonds(m_all_possible_bonds, access_location::device, access_mode::readwrite);

  gpu::sort_and_remove_zeros_possible_bond_array(d_all_possible_bonds.data,
                                       size,
                                       m_num_nonzero_bonds.getDeviceFlags());

  if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
  m_num_all_possible_bonds = m_num_nonzero_bonds.readFlags();
  if (m_prof) m_prof->pop();


  if (m_prof) m_prof->push("filterPossibleBonds2");
  //filter out the existing bonds - based on neighbor list exclusion handeling
  m_tuner_filter_bonds->begin();
  gpu::filter_existing_bonds(d_all_possible_bonds.data,
                             d_n_existing_bonds.data,
                             d_existing_bonds_list.data,
                             m_existing_bonds_list_indexer,
                             m_num_all_possible_bonds,
                             m_tuner_filter_bonds->getParam());
  if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
  m_tuner_filter_bonds->end();
  if (m_prof) m_prof->pop();
  if (m_prof) m_prof->push("filterPossibleBonds3");

  // filtering existing bonds out introduced some zeros back into the array, remove them
  gpu::remove_zeros_possible_bond_array(d_all_possible_bonds.data,
                                       m_num_all_possible_bonds,
                                       m_num_nonzero_bonds.getDeviceFlags());
  if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();

  m_num_all_possible_bonds = m_num_nonzero_bonds.readFlags();

  if (m_prof) m_prof->pop();
// at this point, the sub-array: d_all_possible_bonds[0,m_num_all_possible_bonds]
// should contain only unique entries of possible bonds which are not yet formed.
  }



/*!
 * (Re-)computes the translation vectors for traversing the BVH tree. At most, there are 26 translation vectors
 * when the simulation box is 3D periodic (the self-image is excluded). In 2D, there are at most 8 translation vectors.
 * In MPI runs, a ghost layer of particles is added from adjacent ranks, so there is no need to perform any translations
 * in this direction. The translation vectors are determined by linear combination of the lattice vectors, and must be
 * recomputed any time that the box resizes.
 */
void DynamicBondUpdaterGPU::updateImageVectors()
    {

    const BoxDim& box = m_pdata->getBox();
    uchar3 periodic = box.getPeriodic();
    unsigned char sys3d = (m_sysdef->getNDimensions() == 3);

    // now compute the image vectors
    // each dimension increases by one power of 3
    unsigned int n_dim_periodic = (periodic.x + periodic.y + sys3d*periodic.z);
    m_n_images = 1;
    for (unsigned int dim = 0; dim < n_dim_periodic; ++dim)
        {
        m_n_images *= 3;
        }
    m_n_images -= 1; // remove the self image

    // reallocate memory if necessary
    if (m_n_images > m_image_list.getNumElements())
        {
        GlobalVector<Scalar3> image_list(m_n_images, m_exec_conf);
        m_image_list.swap(image_list);
        }

    ArrayHandle<Scalar3> h_image_list(m_image_list, access_location::host, access_mode::overwrite);
    Scalar3 latt_a = box.getLatticeVector(0);
    Scalar3 latt_b = box.getLatticeVector(1);
    Scalar3 latt_c = box.getLatticeVector(2);

    // iterate over all other combinations of images, skipping those that are
    unsigned int n_images = 0;
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

                    h_image_list.data[n_images] = Scalar(i) * latt_a + Scalar(j) * latt_b + Scalar(k) * latt_c;
                    ++n_images;
                    }
                }
            }
        }

    }

namespace detail
{
/*!
 * \param m Python module to export to
 */
 void export_DynamicBondUpdaterGPU(pybind11::module& m)
     {
     namespace py = pybind11;
     py::class_< DynamicBondUpdaterGPU, std::shared_ptr<DynamicBondUpdaterGPU> >(m, "DynamicBondUpdaterGPU", py::base<DynamicBondUpdater>())
         .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, std::shared_ptr<ParticleGroup>,
              std::shared_ptr<ParticleGroup>, const Scalar, const Scalar, unsigned int, unsigned int, unsigned int>());

         //.def_property("inside", &DynamicBondUpdater::getInsideType, &DynamicBondUpdater::setInsideType)
         //.def_property("outside", &DynamicBondUpdater::getOutsideType, &DynamicBondUpdater::setOutsideType)
         //.def_property("lo", &DynamicBondUpdater::getRegionLo, &DynamicBondUpdater::setRegionLo)
         //.def_property("hi", &DynamicBondUpdater::getRegionHi, &DynamicBondUpdater::setRegionHi);
     }

} // end namespace detail
} // end namespace azplugins
