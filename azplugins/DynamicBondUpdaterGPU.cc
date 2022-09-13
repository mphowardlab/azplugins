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

/*!
 * \param sysdef System definition
 *
 * The system is initialized in a configuration that will be invalid on the
 * first check of the types and region. This constructor requires that the user
 * properly initialize the system via setters.
 */
DynamicBondUpdaterGPU::DynamicBondUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                                            std::shared_ptr<ParticleGroup> group_1,
                                            std::shared_ptr<ParticleGroup> group_2,
                                            unsigned int seed)
        : DynamicBondUpdater(sysdef, group_1, group_2, seed),  m_num_nonzero_bonds_flag(m_exec_conf), m_max_bonds_overflow_flag(m_exec_conf),
          m_lbvh(m_exec_conf), m_traverser(m_exec_conf)
    {
    m_exec_conf->msg->notice(5) << "Constructing DynamicBondUpdaterGPU" << std::endl;

    m_copy_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "dynamic_bond_copy", m_exec_conf));
    m_copy_nlist_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "dynamic_bond_nlist_copy", m_exec_conf));
    m_tuner_filter_bonds.reset(new Autotuner(32, 1024, 32, 5, 100000, "dynamic_bond_filter_bonds", m_exec_conf));
    }

DynamicBondUpdaterGPU::DynamicBondUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                                              std::shared_ptr<NeighborList> pair_nlist,
                                              std::shared_ptr<ParticleGroup> group_1,
                                              std::shared_ptr<ParticleGroup> group_2,
                                              const Scalar r_cut,
                                              const Scalar probability,
                                              unsigned int bond_type,
                                              unsigned int max_bonds_group_1,
                                              unsigned int max_bonds_group_2,
                                              unsigned int seed)
        : DynamicBondUpdater(sysdef, pair_nlist, group_1, group_2,
                            r_cut, probability, bond_type, max_bonds_group_1, max_bonds_group_2,seed),
        m_num_nonzero_bonds_flag(m_exec_conf), m_max_bonds_overflow_flag(m_exec_conf),
        m_lbvh(m_exec_conf), m_traverser(m_exec_conf)
    {
    m_exec_conf->msg->notice(5) << "Constructing DynamicBondUpdaterGPU" << std::endl;

    m_copy_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "dynamic_bond_copy", m_exec_conf));
    m_copy_nlist_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "dynamic_bond_nlist_copy", m_exec_conf));
    m_tuner_filter_bonds.reset(new Autotuner(32, 1024, 32, 5, 100000, "dynamic_bond_filter_bonds", m_exec_conf));

    }

DynamicBondUpdaterGPU::~DynamicBondUpdaterGPU()
    {
      m_exec_conf->msg->notice(5) << "Destroying DynamicBondUpdaterGPU" << std::endl;
    }


void DynamicBondUpdaterGPU::buildTree()
    {

    if (m_prof) m_prof->push("buildTree");

    ArrayHandle<unsigned int> d_index_group_2(m_group_2->getIndexArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    const BoxDim lbvh_box = getLBVHBox();

    // build a lbvh for group_2
    // this tree is traversed in traverseTree()
    m_lbvh.build(azplugins::gpu::PointMapInsertOp(d_pos.data,
                                d_index_group_2.data,
                                m_group_2->getNumMembers()),
                                lbvh_box.getLo(),
                                lbvh_box.getHi());


   if (m_prof) m_prof->pop();
   }

void DynamicBondUpdaterGPU::traverseTree()
    {
   if (m_prof) m_prof->push("traverseTree");
    ArrayHandle<unsigned int> d_nlist(m_n_list, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_n_neigh(m_n_neigh, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

    // clear the neighbor counts
    cudaMemset(d_n_neigh.data,0, sizeof(unsigned int)*m_group_1->getNumMembers());

    const BoxDim& box = m_pdata->getBox();

    // neighbor list write op
    azplugins::gpu::NeighborListOp nlist_op(d_nlist.data, d_n_neigh.data, m_max_bonds_overflow_flag.getDeviceFlags(), m_max_bonds);

    ArrayHandle<unsigned int> d_index_group_1(m_group_1->getIndexArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_index_group_2(m_group_2->getIndexArray(), access_location::device, access_mode::read);

    neighbor::MapTransformOp map(d_index_group_2.data );
    m_traverser.setup(map, m_lbvh);

    azplugins::gpu::ParticleQueryOp query_op(d_pos.data,
                                               d_index_group_1.data,
                                               m_group_1->getNumMembers(),
                                               m_pdata->getMaxN(),
                                               m_r_cut,
                                               box);

     m_traverser.traverse(nlist_op, query_op, map,m_lbvh, m_image_list);

     m_max_bonds_overflow =  m_max_bonds_overflow_flag.readFlags();

    if (m_prof) m_prof->pop();

    }



void DynamicBondUpdaterGPU::filterPossibleBonds()
   {

    if (m_prof) m_prof->push("filterPossibleBonds copy ");
    //copy data from m_n_list to d_all_possible_bonds. nlist saves indices and the existing bonds have to be stored by tags
    //so copy data first then sort out existing bonds
    const unsigned int size = m_group_1->getNumMembers()*m_max_bonds;
    ArrayHandle<unsigned int> d_n_existing_bonds(m_n_existing_bonds, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_existing_bonds_list(m_existing_bonds_list, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nlist(m_n_list, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_n_neigh(m_n_neigh, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_index_group_1(m_group_1->getIndexArray(), access_location::device, access_mode::read);

    ArrayHandle<Scalar3> d_all_possible_bonds(m_all_possible_bonds, access_location::device, access_mode::readwrite);
    cudaMemset((void*)d_all_possible_bonds.data, 0, sizeof(Scalar3)*m_all_possible_bonds.getNumElements());

    const BoxDim& box = m_pdata->getBox();

    m_copy_tuner->begin();
    gpu::copy_possible_bonds(d_all_possible_bonds.data,
                              d_pos.data,
                              d_tag.data,
                              d_index_group_1.data,
                              d_n_neigh.data,
                              d_nlist.data,
                              box,
                              m_max_bonds,
                              m_r_cut,
                              m_groups_identical,
                              m_group_1->getNumMembers(),
                              m_copy_tuner->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_copy_tuner->end();

    if (m_prof) m_prof->pop();
    if (m_prof) m_prof->push("filterPossibleBonds remove existing");

    //filter out the existing bonds - based on neighbor list exclusion handeling
    m_tuner_filter_bonds->begin();
    gpu::filter_existing_bonds(d_all_possible_bonds.data,
                               d_n_existing_bonds.data,
                               d_existing_bonds_list.data,
                               m_existing_bonds_list_indexer,
                               size,
                               m_tuner_filter_bonds->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tuner_filter_bonds->end();

    if (m_prof) m_prof->pop();
    if (m_prof) m_prof->push("filterPossibleBonds sort_remove");

    m_num_all_possible_bonds = 0;

    gpu::remove_zeros_and_sort_possible_bond_array(d_all_possible_bonds.data,
                                                   size,
                                                   m_num_nonzero_bonds_flag.getDeviceFlags());

    m_num_all_possible_bonds = m_num_nonzero_bonds_flag.readFlags();

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
       .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, std::shared_ptr<ParticleGroup>, unsigned int>())
       .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, std::shared_ptr<ParticleGroup>,
              std::shared_ptr<ParticleGroup>, Scalar, Scalar, unsigned int, unsigned int, unsigned int, unsigned int>());
     }

} // end namespace detail
} // end namespace azplugins
