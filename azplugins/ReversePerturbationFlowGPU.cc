// Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file ReversePerturbationFlowGPU.cc
 * \brief Definition of ReversePerturbationFlowGPU
 */

#include "ReversePerturbationFlowGPU.h"
#include "ReversePerturbationFlowGPU.cuh"

namespace azplugins
{

/*!
 * \param sysdef SystemDefinition this updater will act on
 * \param group Group to operate on
 * \param num_swap Max number of swaps
 * \param slab_width Slab thickness
 * \param p_target target momentum for swapping pairs
 */
ReversePerturbationFlowGPU::ReversePerturbationFlowGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                       std::shared_ptr<ParticleGroup> group,
                                                       unsigned int num_swap,
                                                       Scalar slab_width,
                                                       Scalar p_target)
        : ReversePerturbationFlow(sysdef, group, num_swap, slab_width, p_target),
         m_num_mark(m_exec_conf), m_split(m_exec_conf), m_type(m_exec_conf)
    {
    m_swap_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "rp_swap_velocities", m_exec_conf));
    m_mark_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "rp_mark_particles", m_exec_conf));
    m_fill_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "rp_fill_pair_array", m_exec_conf));
    m_split_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "rp_split_pair_array", m_exec_conf));

    GPUArray<Scalar2> pairs(m_group->getNumMembers(), m_pdata->getExecConf());
    m_slab_pairs.swap(pairs);
    }

/*!
 *  1. All particles are marked with a flag, depending in which slab they are in
 *     either = 0 if in no slab, +(index+1) if in bottom slab, -(index-1) for top
 *     slab
 *  2. Select all entries with flags =/= 0
 *  3. Sort those entries according to sign of flags and momentum.
 *  4. Determine how many entries are in both slabs and how many are in top and bottom
 *     slab.
 *  5. Seperate and rearange the top and bottom slab entries into two seperate arrays,
 *     each are sorted according to their absolute momentum
 */
void ReversePerturbationFlowGPU::findSwapParticles()
    {
    if(m_num_swap==0)// No swaps to perform
        {
        m_split.resetFlags(0);
        m_num_lo = 0;
        m_num_hi = 0;
        return;
        }
    // check if group size changed
    if (m_group->getNumMembers() > m_slab_pairs.getNumElements())
        {
        GPUArray<Scalar2> slab_pairs(m_group->getNumMembers(), m_exec_conf);
        m_slab_pairs.swap(slab_pairs);
        }

    ArrayHandle<Scalar2> d_slab_pairs(m_slab_pairs, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_member_idx(m_group->getIndexArray(), access_location::device, access_mode::read);

    /* mark candidate particles in slabs (d_select_flags = 0 for particles outside either slab
    *  d_select_flags = -(index+1) for top slab, and +(index+1) for bottom slab)
    *  needs momentum because only the ones moving in the right direction are picked
    *  bottom slab: positive momentum in x-direction
    *  top slab: negative momentum in x-direction
    */
    m_mark_tuner->begin();
    gpu::mark_particles_in_slabs(d_slab_pairs.data,
                                 d_pos.data,
                                 d_vel.data,
                                 d_member_idx.data,
                                 m_lo_pos,
                                 m_hi_pos,
                                 m_group->getNumMembers(),
                                 m_mark_tuner->getParam());
    m_mark_tuner->end();
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    // use cub device select to filter out the marked particles
    // every particle with d_select_flags !=0 is selected
    // needs to be called twice, first time for memory allocation
    void *d_tmp_storage = NULL;
    size_t tmp_storage_bytes = 0;
    gpu::select_particles_in_slabs(m_num_mark.getDeviceFlags(),
                                   d_tmp_storage,
                                   tmp_storage_bytes,
                                   d_slab_pairs.data,
                                   m_group->getNumMembers());

    size_t alloc_size = (tmp_storage_bytes > 0) ? tmp_storage_bytes : 4;
    ScopedAllocation<char> d_alloc(m_exec_conf->getCachedAllocator(), alloc_size);
    d_tmp_storage = (void *)d_alloc();

    gpu::select_particles_in_slabs(m_num_mark.getDeviceFlags(),
                                   d_tmp_storage,
                                   tmp_storage_bytes,
                                   d_slab_pairs.data,
                                   m_group->getNumMembers());

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    const unsigned int Nslab = m_num_mark.readFlags();

    // sort pairs based on their momentum. also automatically sorts by slab
    // m_slab_pairs now contains a list of [+/- (index+1), momentum] sorted by
    // first top slab, momentum closest to -p_target, to bottom slab, momentum closest to +p_target
    gpu::sort_pair_array(d_slab_pairs.data,Nslab,m_p_target);

    /* for splitting the array m_slab_pairs into two parts for bottom and top
     * slab, the information how many entries there are in each part is needed
     * beforehand. The next part of the code determines how many particles are in
     * each slab and if there are zero in either layer determines which one is zero.
     */
    ArrayHandle<Scalar2> d_layer_hi(m_layer_hi, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar2> d_layer_lo(m_layer_lo, access_location::device, access_mode::overwrite);

    m_split.resetFlags(Nslab);
    m_type.resetFlags(0);

    m_split_tuner->begin();
    // find out how many particles are in each layer and if it's just top or bottom if no split is found
    gpu::find_split_array(m_split.getDeviceFlags(),
                          m_type.getDeviceFlags(),
                          d_slab_pairs.data,
                          Nslab,
                          m_mark_tuner->getParam());
    m_split_tuner->end();
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    unsigned int split = m_split.readFlags();
    /* this part assignes the right values to m_num_hi and m_num_lo depending
    *  on the result of gpu::find_split_array. If a split in the array is found,
    *  there are entries in both top and bottom slab, if not the m_type flag is used
    *  to determine whether the top or the bottom is empty.
    */
    if (split == Nslab) // no split found
        {
        if (m_type.readFlags()<0) // first element has a negative tag -> all entries are top slab
            {
            m_num_hi = Nslab;
            m_num_lo = 0;
            }
        else   // first element has a positive tag -> all entries are bottom slab
            {
            m_num_hi = 0;
            m_num_lo = Nslab;
            }
        }
    else // split found
        {
        m_num_hi = split;
        m_num_lo = Nslab-split;
        }

    unsigned int num_lo_entries = std::min(m_num_swap,m_num_lo);
    unsigned int num_hi_entries = std::min(m_num_swap,m_num_hi);
    unsigned int num_threads = num_lo_entries +  num_hi_entries;

    m_fill_tuner->begin();
    // m_slab_pairs needs to be devided into the two arrays m_layer_hi and m_layer_lo
    gpu::divide_pair_array(d_slab_pairs.data,
                           d_layer_hi.data,
                           d_layer_lo.data,
                           num_hi_entries,
                           Nslab,
                           num_threads,
                           m_mark_tuner->getParam());
    m_fill_tuner->end();
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }

/*!
 *  6. Take two arrays for top and bottom slab and  swap the momentum of up to num_pairs pairs
 *  7. calculate the total momentum exchange
 */
void ReversePerturbationFlowGPU::swapPairMomentum()
    {
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar2> d_layer_hi(m_layer_hi, access_location::device, access_mode::read);
    ArrayHandle<Scalar2> d_layer_lo(m_layer_lo, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_member_idx(m_group->getIndexArray(), access_location::device, access_mode::read);
    unsigned int num_pairs = std::min(m_num_swap, std::min(m_num_lo, m_num_hi));

    m_swap_tuner->begin();
    gpu::swap_momentum_pairs(d_layer_hi.data,
                             d_layer_lo.data,
                             d_vel.data,
                             d_member_idx.data,
                             num_pairs,
                             m_swap_tuner->getParam());

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_swap_tuner->end();

    m_momentum_exchange =gpu::calc_momentum_exchange(d_layer_hi.data, d_layer_lo.data, num_pairs);
    }

namespace detail
{
/*!
 * \param m Python module to export to
 */
void export_ReversePerturbationFlowGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_< ReversePerturbationFlowGPU, std::shared_ptr<ReversePerturbationFlowGPU> >(m, "ReversePerturbationFlowGPU", py::base<ReversePerturbationFlow>())
        .def(py::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<ParticleGroup>, unsigned int, Scalar,Scalar>());
    }
} // end namespace detail
} // end namespace azplugins
