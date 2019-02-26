// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file ParticleEvaporatorGPU.cc
 * \brief Definition of ParticleEvaporatorGPU
 */

#include "ParticleEvaporatorGPU.h"
#include "ParticleEvaporatorGPU.cuh"

namespace azplugins
{

/*!
 * \param sysdef System definition
 * \param seed Seed to the pseudo-random number generator
 *
 * The system is initialized in a configuration that will be invalid on the
 * first check of the types and region. This constructor requires that the user
 * properly initialize the system via setters.
 */
ParticleEvaporatorGPU::ParticleEvaporatorGPU(std::shared_ptr<SystemDefinition> sysdef, unsigned int seed)
        : ParticleEvaporator(sysdef, seed), m_select_flags(m_exec_conf), m_num_mark(m_exec_conf)
    {
    m_mark_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "evap_mark_particles", m_exec_conf));
    m_pick_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "evap_pick_particles", m_exec_conf));
    }

/*!
 * \param sysdef System definition
 * \param inside_type Type id of particles inside region
 * \param outside_type Type id of particles outside region
 * \param z_lo Lower bound of region in z
 * \param z_hi Upper bound of region in z
 * \param seed Seed to the pseudo-random number generator
 */
ParticleEvaporatorGPU::ParticleEvaporatorGPU(std::shared_ptr<SystemDefinition> sysdef,
                               unsigned int inside_type,
                               unsigned int outside_type,
                               Scalar z_lo,
                               Scalar z_hi,
                               unsigned int seed)
        : ParticleEvaporator(sysdef, inside_type, outside_type, z_lo, z_hi, seed),
          m_select_flags(m_exec_conf), m_num_mark(m_exec_conf)
    {
    m_mark_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "evap_mark_particles", m_exec_conf));
    m_pick_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "evap_pick_particles", m_exec_conf));
    }

/*!
 * Particles are marked for evaporation in two phases. First, candidate particles
 * are marked in gpu::evaporate_setup_mark. Then, the CUB library is used to select
 * the indexes of these marked particles as possible picks in a compacted array.
 * This method is less memory-efficient than the CPU implementation (marking
 * particles requires O(N) memory rather than O(Npick)), but still seems
 * to be the best implementation for the GPU.
 */
unsigned int ParticleEvaporatorGPU::markParticles()
    {
    if (m_pdata->getN() == 0)
        return 0;

    m_mark.resize(m_pdata->getN());
    m_select_flags.resize(m_pdata->getN());
    m_num_mark.resetFlags(0);

    ArrayHandle<unsigned char> d_select_flags(m_select_flags, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_mark(m_mark, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

    // mark candidate particles for evaporation
    m_mark_tuner->begin();
    gpu::evaporate_setup_mark(d_select_flags.data,
                                   d_mark.data,
                                   d_pos.data,
                                   m_outside_type,
                                   m_z_lo,
                                   m_z_hi,
                                   m_pdata->getN(),
                                   m_mark_tuner->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_mark_tuner->end();

    // use cub device select to filter out the marked particles
        {
        void *d_tmp_storage = NULL;
        size_t tmp_storage_bytes = 0;
        gpu::evaporate_select_mark(d_mark.data,
                                   m_num_mark.getDeviceFlags(),
                                   d_tmp_storage,
                                   tmp_storage_bytes,
                                   d_select_flags.data,
                                   m_pdata->getN());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        size_t alloc_size = (tmp_storage_bytes > 0) ? tmp_storage_bytes : 4;
        ScopedAllocation<unsigned char> d_alloc(m_exec_conf->getCachedAllocator(), alloc_size);
        d_tmp_storage = (void *)d_alloc();

        gpu::evaporate_select_mark(d_mark.data,
                                   m_num_mark.getDeviceFlags(),
                                   d_tmp_storage,
                                   tmp_storage_bytes,
                                   d_select_flags.data,
                                   m_pdata->getN());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    const unsigned int N_mark = m_num_mark.readFlags();
    return N_mark;
    }

/*!
 * Picks are applied on the GPU. Typically, the number of picks is small, so this
 * kernel call will have significant overhead. However, it should typically still
 * be faster than copying particle positions from device to host and back.
 * This performance should be monitored in profiling if evaporation is noticed to
 * be slow.
 */
void ParticleEvaporatorGPU::applyPicks()
    {
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_picks(m_picks, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_mark(m_mark, access_location::device, access_mode::read);

    m_pick_tuner->begin();
    gpu::evaporate_apply_picks(d_pos.data,
                               d_picks.data,
                               d_mark.data,
                               m_inside_type,
                               m_Npick,
                               m_pick_tuner->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_pick_tuner->end();
    }

namespace detail
{
/*!
 * \param m Python module to export to
 */
void export_ParticleEvaporatorGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_< ParticleEvaporatorGPU, std::shared_ptr<ParticleEvaporatorGPU> >(m, "ParticleEvaporatorGPU", py::base<ParticleEvaporator>())
        .def(py::init<std::shared_ptr<SystemDefinition>, unsigned int>())
        .def(py::init<std::shared_ptr<SystemDefinition>, unsigned int, unsigned int, Scalar, Scalar, unsigned int>());
    }
} // end namespace detail

} // end namespace azplugins
