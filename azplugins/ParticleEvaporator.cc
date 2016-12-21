// Copyright (c) 2016, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file ParticleEvaporator.h
 * \brief Definition of ParticleEvaporator
 */

#include "ParticleEvaporator.h"
#include "hoomd/extern/saruprng.h"
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
    : TypeUpdater(sysdef), m_seed(seed), m_Nevap_max(0xffffffff)
    {
    GPUVector<unsigned int> mark(m_exec_conf);
    m_mark.swap(mark);
    }

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
          m_seed(seed), m_Nevap_max(0xffffffff)
    {
    GPUVector<unsigned int> mark(m_exec_conf);
    m_mark.swap(mark);
    }

/*!
 * \param timestep Timestep update is called
 */
void ParticleEvaporator::changeTypes(unsigned int timestep)
    {
    if (m_prof) m_prof->push("evaporate");

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

    // select particles for deletion
    m_picks.clear();
    if (N_mark_total < m_Nevap_max)
        {
        // fill the picks up with all of the particles this rank owns
        m_picks.resize(N_mark);
        std::iota(m_picks.begin(), m_picks.end(), 0);
        }
    else
        {
        // fill up vector which we will randomly shuffle
        std::vector<unsigned int> global_marks(N_mark_total);
        std::iota(global_marks.begin(), global_marks.end(), 0);

        // random shuffle (fisher-yates) to get picks, seeded the same across
        // all ranks to yield identical choices from integer math
            {
            Saru rng(m_seed, timestep);

            auto begin = global_marks.begin();
            auto end = global_marks.end();
            size_t left = std::distance(begin,end);
            unsigned int N_pick = m_Nevap_max;
            while (N_pick--)
                {
                auto r = begin;
                std::advance(r, rng.u32() % left);
                std::swap(*begin, *r);
                ++begin;
                --left;
                }
            }

        // select the picks that lie on my rank, with reindexing to local mark indexes
        const unsigned int max_pick_idx = N_before + N_mark;
        for (unsigned int i=0; i < m_Nevap_max; ++i)
            {
            const unsigned int pick = global_marks[i];
            if (pick >= N_before && pick < max_pick_idx)
                {
                m_picks.push_back(pick - N_before);
                }
            }
        }

    std::cout << m_picks.size() << std::endl;

    applyPicks();

    // each rank applies the actual evaporation

    if (m_prof) m_prof->pop();
    }

unsigned int ParticleEvaporator::markParticles()
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
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
    ArrayHandle<unsigned int> h_mark(m_mark, access_location::host, access_mode::read);

    for (unsigned int i=0; i < m_picks.size(); ++i)
        {
        const unsigned int pidx = h_mark.data[m_picks[i]];
        h_pos.data[pidx].w = __int_as_scalar(m_inside_type);
        }
    }

namespace detail
{
//! Export the ParticleEvaporator to python
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
