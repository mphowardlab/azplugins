// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file ParticleEvaporator.h
 * \brief Definition of ParticleEvaporator
 */

#include "ParticleEvaporator.h"
#include "hoomd/RandomNumbers.h"
#include "RNGIdentifiers.h"
#include <algorithm>

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
ParticleEvaporator::ParticleEvaporator(std::shared_ptr<SystemDefinition> sysdef, unsigned int seed)
    : TypeUpdater(sysdef), m_seed(seed), m_Nevap_max(0xffffffff), m_Npick(0), m_picks(m_exec_conf), m_mark(m_exec_conf)
    {}

/*!
 * \param sysdef System definition
 * \param inside_type Type id of particles inside region
 * \param outside_type Type id of particles outside region
 * \param z_lo Lower bound of region in z
 * \param z_hi Upper bound of region in z
 * \param seed Seed to the pseudo-random number generator
 */
ParticleEvaporator::ParticleEvaporator(std::shared_ptr<SystemDefinition> sysdef,
                                       unsigned int inside_type,
                                       unsigned int outside_type,
                                       Scalar z_lo,
                                       Scalar z_hi,
                                       unsigned int seed)
        : TypeUpdater(sysdef, inside_type, outside_type, z_lo, z_hi),
          m_seed(seed), m_Nevap_max(0xffffffff), m_Npick(0), m_picks(m_exec_conf), m_mark(m_exec_conf)
    {}

/*!
 * \param timestep Timestep update is called
 *
 * Particle evaporation proceeds in four steps:
 *  1. Mark all possible particles for evaporation on the local rank, and compact their particle
 *     indexes into an array running from 0 to \a N_mark.
 *  2. Perform an MPI exclusive scan and reduction to determine the total number of marked particles on all
 *     ranks and the global marked indexes that the rank owns.
 *  3. Randomly choose up to m_Nevap_max particles from the global list. Each rank performs this task
 *     independently using the same PRNG with the same seed. This guarantees the selection of the same
 *     particles, and avoids any extra communication.
 *  4. The random picks are applied by flipping the types of the particles.
 */
void ParticleEvaporator::changeTypes(unsigned int timestep)
    {
    // to avoid having to divy up profiling into cuda functions when virtual
    // functions are subclassed, use a switch here to setup the right profiling environment
    if (m_prof)
        {
        if (m_exec_conf->isCUDAEnabled())
            {
            m_prof->push(m_exec_conf, "evaporate");
            }
        else
            {
            m_prof->push("evaporate");
            }
        }

    // mark particles as candidates for evaporation
    bool overflowed = false;
    unsigned int N_mark = 0;
    do
        {
        N_mark = markParticles();
        overflowed = (N_mark > m_mark.size());
        if (overflowed)
            {
            m_mark.resize(N_mark);
            }
        } while(overflowed);

    // reduce / scan the number of particles that are marked on all ranks
    unsigned int N_mark_total = N_mark;
    unsigned int N_before = 0;
    #ifdef ENABLE_MPI
    if (m_exec_conf->getNRanks() > 1)
        {
        MPI_Allreduce(&N_mark, &N_mark_total, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Exscan(&N_mark, &N_before, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif // ENABLE_MPI

    // pick which particles will be evaporated
    if (N_mark_total < m_Nevap_max) // bypass any picking logic, and take all particles
        {
        m_picks.resize(N_mark);
        ArrayHandle<unsigned int> h_picks(m_picks, access_location::host, access_mode::overwrite);
        std::iota(h_picks.data, h_picks.data + N_mark, 0);
        m_Npick = N_mark;
        }
    else // do the more complicated random selection
        {
        makeAllPicks(timestep, m_Nevap_max, N_mark_total);

        /*
         * Select the picks that lie on my rank, with reindexing to local mark indexes.
         * This is performed in a do loop to allow for resizing of the GPUVector.
         * After a short time, the loop will be ignored.
         */
        const unsigned int max_pick_idx = N_before + N_mark;
        overflowed = false;
        do
            {
            m_Npick = 0;
            const unsigned int max_Npick = m_picks.size();

                {
                ArrayHandle<unsigned int> h_picks(m_picks, access_location::host, access_mode::overwrite);
                for (unsigned int i=0; i < m_Nevap_max; ++i)
                    {
                    const unsigned int pick = m_all_picks[i];
                    if (pick >= N_before && pick < max_pick_idx)
                        {
                        if (m_Npick < max_Npick)
                            {
                            h_picks.data[m_Npick] = pick - N_before;
                            }
                        ++m_Npick;
                        }
                    }
                }

            overflowed = (m_Npick > max_Npick);
            if (overflowed)
                {
                m_picks.resize(m_Npick);
                }

            } while (overflowed);
        }

    // each rank applies the evaporation to the particles
    applyPicks();

    if (m_prof)
        {
        if (m_exec_conf->isCUDAEnabled())
            {
            m_prof->pop(m_exec_conf);
            }
        else
            {
            m_prof->pop();
            }
        }
    }

unsigned int ParticleEvaporator::markParticles()
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_mark(m_mark, access_location::host, access_mode::overwrite);

    unsigned int N_mark = 0;
    const unsigned int N_mark_max = m_mark.size();
    for (unsigned int i=0; i < m_pdata->getN(); ++i)
        {
        const Scalar4 postype = h_pos.data[i];
        const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        unsigned int type = __scalar_as_int(postype.w);

        // only check particles that are of the "outside" (solvent) type
        if (type == m_outside_type)
            {
            // test for overlap as for an AABB
            bool inside = !(pos.z > m_z_hi || pos.z < m_z_lo);
            if (inside)
                {
                if (N_mark < N_mark_max) h_mark.data[N_mark] = i;
                ++N_mark;
                }
            }
        }
    return N_mark;
    }

void ParticleEvaporator::applyPicks()
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_picks(m_picks, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_mark(m_mark, access_location::host, access_mode::read);

    for (unsigned int i=0; i < m_Npick; ++i)
        {
        const unsigned int pidx = h_mark.data[h_picks.data[i]];
        h_pos.data[pidx].w = __int_as_scalar(m_inside_type);
        }
    }

/*!
 * \param timestep Current timestep
 * \param N_pick Number of particles to pick
 * \param N_mark_total Total number of particles marked for picking
 *
 * The Fisher-Yates shuffle algorithm is applied to randomly pick unique particles
 * out of the possible particles across all ranks. The result is stored in
 * \a m_all_picks.
 */
void ParticleEvaporator::makeAllPicks(unsigned int timestep, unsigned int N_pick, unsigned int N_mark_total)
    {
    assert(N_pick <= N_mark_total);

    // fill up vector which we will randomly shuffle
    m_all_picks.resize(N_mark_total);
    std::iota(m_all_picks.begin(), m_all_picks.end(), 0);

    hoomd::RandomGenerator rng(azplugins::RNGIdentifiers::ParticleEvaporator, m_seed, timestep);

    // random shuffle (fisher-yates) to get picks, seeded the same across all ranks
    auto begin = m_all_picks.begin();
    auto end = m_all_picks.end();
    size_t left = std::distance(begin,end);
    unsigned int N_choose = N_pick;
    while (N_choose-- && left > 1)
        {
        hoomd::UniformIntDistribution rand_shift(left-1);

        auto r = begin;
        std::advance(r, rand_shift(rng));
        std::swap(*begin, *r);
        ++begin;
        --left;
        }

    // size the vector down to the number picked
    m_all_picks.resize(N_pick);
    }

namespace detail
{
/*!
 * \param m Python module to export to
 */
void export_ParticleEvaporator(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_< ParticleEvaporator, std::shared_ptr<ParticleEvaporator> >(m, "ParticleEvaporator", py::base<TypeUpdater>())
        .def(py::init<std::shared_ptr<SystemDefinition>, unsigned int>())
        .def(py::init<std::shared_ptr<SystemDefinition>, unsigned int, unsigned int, Scalar, Scalar, unsigned int>())
        .def_property("Nmax", &ParticleEvaporator::getNEvapMax, &ParticleEvaporator::setNEvapMax);
    }
} // end namespace detail

} // end namespace azplugins
