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
                          std::shared_ptr<ParticleGroup> group_1,
                          std::shared_ptr<ParticleGroup> group_2,
                          const Scalar r_cut,
                          unsigned int bond_type,
                          unsigned int max_bonds_group_1,
                          unsigned int max_bonds_group_2)
        : DynamicBondUpdater(sysdef, group_1, group_2, r_cut, bond_type, max_bonds_group_1, max_bonds_group_2),
        m_num_nonzero_bonds(m_exec_conf), m_lbvh(m_exec_conf), m_traverser(m_exec_conf)
    {

    m_tuner_filter_bonds.reset(new Autotuner(32, 1024, 32, 5, 100000, "dynamic_bond_updater_filter_bonds", m_exec_conf));
    m_copy_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "dynamic_bond_updater_tree_copy", m_exec_conf));

    }

DynamicBondUpdaterGPU::~DynamicBondUpdaterGPU()
    {
    }

// this is based on the LBVH NeighborListGPUTree implementation
void DynamicBondUpdaterGPU::findAllPossibleBonds()
    {
      std::cout<< "in DynamicBondUpdaterGPU::findAllPossibleBonds"<<std::endl;

      const BoxDim& box = m_pdata->getBox();

      ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::read);
      ArrayHandle<unsigned int> d_group_1_indexes(m_group_1->getIndexArray(), access_location::device, access_mode::read);

      unsigned int group_size_1 = m_group_1->getNumMembers();
      const BoxDim lbvh_box = getLBVHBox();

      // build a lbvh for group_1
      m_lbvh.build(gpu::PointMapInsertOp(d_postype.data, d_group_1_indexes.data, group_size_1),
                                lbvh_box.getLo(),
                                lbvh_box.getHi());

      //todo: need to use the neighbor list data structure, unclear how to restructure the internal cast into uint4 otherwise?
      GPUArray<unsigned int> n_bonds(m_group_2->getNumMembers(), m_exec_conf);
      ArrayHandle<unsigned int> h_n_bonds(n_bonds, access_location::host, access_mode::overwrite);
      memset((void*)h_n_bonds.data, 0, sizeof(unsigned int)*m_group_2->getNumMembers());

      ArrayHandle<unsigned int> d_n_bonds(n_bonds, access_location::device, access_mode::readwrite);

      unsigned int size = m_group_2->getNumMembers()*m_max_bonds;
      GPUArray<unsigned int> all_possible_bonds_int(size, m_exec_conf);
      ArrayHandle<unsigned int> h_all_possible_bonds_int(all_possible_bonds_int, access_location::host, access_mode::overwrite);
      memset((void*)h_all_possible_bonds_int.data, 0, sizeof(unsigned int)*size);

      ArrayHandle<unsigned int> d_all_possible_bonds_int(all_possible_bonds_int, access_location::device, access_mode::readwrite);

      GPUArray<unsigned int> head_list(m_group_2->getNumMembers(), m_exec_conf);
      ArrayHandle<unsigned int> h_head_list(head_list, access_location::host, access_mode::overwrite);

      unsigned int headAddress = 0;
      for (unsigned int i=0; i < m_group_2->getNumMembers(); ++i)
          {
          h_head_list.data[i] = headAddress;
          headAddress+=m_max_bonds;
        }

      ArrayHandle<unsigned int> d_head_list(head_list, access_location::device, access_mode::readwrite);
      // neighbor list write op

      gpu::NeighborListOp nlist_op(d_all_possible_bonds_int.data, d_n_bonds.data,&m_max_bonds_overflow, d_head_list.data, m_max_bonds);

    /*  NeighborListOp(unsigned int* neigh_list_,
                     unsigned int* nneigh_,
                     unsigned int* new_max_neigh_,
                     const unsigned int* first_neigh_,
                     unsigned int max_neigh_)
                     */

      ArrayHandle<unsigned int> d_group_2_indexes(m_group_2->getIndexArray(), access_location::device, access_mode::read);

      gpu::ParticleQueryOp<false,false> query_op(d_postype.data,
                                                  NULL,
                                                  NULL,
                                                  d_group_2_indexes.data,
                                                  m_group_2->getNumMembers(),
                                                  m_group_1->getNumMembers(),
                                                  m_r_cut,
                                                  m_r_cut+0.4,
                                                  box);

      std::cout<< "m_max_bonds_overflow before "<< m_max_bonds_overflow << " m_max_bonds "<< m_max_bonds <<std::endl;
       //neighbor::MapTransformOp map(d_group_2_indexes.data);
       m_traverser.setup(m_lbvh);
       std::cout<<"num group 2 "<< m_group_2->getNumMembers()<<std::endl;
       std::cout<<"num group 1 "<< m_group_1->getNumMembers()<<std::endl;
       std::cout<<"num total "<< m_pdata->getN()<<std::endl;
       ArrayHandle<unsigned int> h_group_2_indexes(m_group_2->getIndexArray(), access_location::host, access_mode::read);
       std::cout<< "group 2 index ";
       for (unsigned int i=0; i < m_group_2->getNumMembers(); ++i)
           {
             std::cout<< " "<< h_group_2_indexes.data[i] ;
         }
         std::cout<< std::endl;
       m_traverser.traverse(nlist_op, query_op, m_lbvh, m_image_list);

      std::cout<< "m_max_bonds_overflow after "<< m_max_bonds_overflow << " m_max_bonds "<< m_max_bonds <<std::endl;

      /*
       ArrayHandle<unsigned int> h_all_possible_bonds_int(all_possible_bonds_int, access_location::host, access_mode::read);
       ArrayHandle<unsigned int> h_n_bonds(n_bonds, access_location::host, access_mode::read);

       for (unsigned int i = 0; i < m_group_2->getNumMembers(); i++)
       {
          std::cout<<" d_n_bonds  "<< i << " "<< h_n_bonds.data[i] << " all_possible_bonds " ;
         for (unsigned int j = 0; j < m_max_bonds; j++)
         {
         std::cout<< "  "<< h_all_possible_bonds_int.data[i*m_max_bonds+j] << " ";
         }
         std::cout <<std::endl;
       }
        std::cout <<" size "<<  m_group_2->getNumMembers()*m_max_bonds<<std::endl;
        */

       // reset content of possible bond list



      // unsigned int size = m_group_2->getNumMembers()*m_max_bonds;
       ArrayHandle<Scalar3> h_all_possible_bonds(m_all_possible_bonds, access_location::host, access_mode::overwrite);
       memset((void*)h_all_possible_bonds.data, 0, sizeof(Scalar3)*size);
       m_num_all_possible_bonds=0;

    }


void DynamicBondUpdaterGPU::filterPossibleBonds()
  {
    std::cout<< "in DynamicBondUpdaterGPU::filterPossibleBonds()"<< std::endl;
  // todo: figure out in which order the thrust calls are the fastest.
  // is using build in thrust functions the best solution?
  // suspect: sort - remove zeros - unique - filter (which introduces zeros) - remove zeros ?

  const unsigned int size = m_group_2->getNumMembers()*m_max_bonds;

  // sort and remove all existing zeros
  ArrayHandle<unsigned int> d_n_existing_bonds(m_n_existing_bonds, access_location::device, access_mode::read);
  ArrayHandle<unsigned int> d_existing_bonds_list(m_existing_bonds_list, access_location::device, access_mode::read);
  ArrayHandle<Scalar3> d_all_possible_bonds(m_all_possible_bonds, access_location::device, access_mode::readwrite);

  gpu::sort_and_remove_zeros_possible_bond_array(d_all_possible_bonds.data,
                                       size,
                                       m_num_nonzero_bonds.getDeviceFlags());


  m_num_all_possible_bonds = m_num_nonzero_bonds.readFlags();

  //filter out the existing bonds - based on neighbor list exclusion handeling
  m_tuner_filter_bonds->begin();
  gpu::filter_existing_bonds(d_all_possible_bonds.data,
                             d_n_existing_bonds.data,
                             d_existing_bonds_list.data,
                             m_existing_bonds_list_indexer,
                             m_num_all_possible_bonds,
                             m_tuner_filter_bonds->getParam());
  m_tuner_filter_bonds->end();


  // filtering existing bonds out introduced some zeros back into the array, remove them
  gpu::remove_zeros_possible_bond_array(d_all_possible_bonds.data,
                                       m_num_all_possible_bonds,
                                       m_num_nonzero_bonds.getDeviceFlags());


  m_num_all_possible_bonds = m_num_nonzero_bonds.readFlags();

// at this point, the sub-array: h_all_possible_bonds[0,m_num_all_possible_bonds]
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
    std::cout<< "in DynamicBondUpdaterGPU::updateImageVectors()"<<std::endl;
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
         .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>,
              std::shared_ptr<ParticleGroup>, const Scalar, unsigned int, unsigned int, unsigned int>());

         //.def_property("inside", &DynamicBondUpdater::getInsideType, &DynamicBondUpdater::setInsideType)
         //.def_property("outside", &DynamicBondUpdater::getOutsideType, &DynamicBondUpdater::setOutsideType)
         //.def_property("lo", &DynamicBondUpdater::getRegionLo, &DynamicBondUpdater::setRegionLo)
         //.def_property("hi", &DynamicBondUpdater::getRegionHi, &DynamicBondUpdater::setRegionHi);
     }

} // end namespace detail
} // end namespace azplugins
