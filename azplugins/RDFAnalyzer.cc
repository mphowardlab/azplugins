// Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file RDFAnalyzer.cc
 * \brief Definition of RDFAnalyzer
 */

#include "RDFAnalyzer.h"

namespace azplugins
{

/*!
 * \param sysdef System definition
 * \param group_1 Particle group
 * \param group_2 Particle
 * \param rcut Cutoff radius for calculation
 * \param bin_width Width for binning particle distances
 */
RDFAnalyzer::RDFAnalyzer(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<ParticleGroup> group_1,
                         std::shared_ptr<ParticleGroup> group_2,
                         Scalar rcut,
                         Scalar bin_width)
    : Analyzer(sysdef), m_group_1(group_1), m_group_2(group_2), m_rcut(rcut),
      m_bin_width(bin_width), m_duplicates(m_exec_conf), m_num_samples(0)
    {
    m_exec_conf->msg->notice(5) << "Constructing RDFAnalyzer" << std::endl;

    assert(m_rcut > 0.0);
    assert(m_bin_width > 0.0);

    // allocate memory for the bins
    m_num_bins = std::ceil(m_rcut / m_bin_width);
    GPUArray<unsigned int> counts(m_num_bins, m_exec_conf);
    m_counts.swap(counts);
    GPUArray<double> accum_rdf(m_num_bins, m_exec_conf);
    m_accum_rdf.swap(accum_rdf);
    reset();

#ifdef ENABLE_MPI
    // MPI is currently not supported due to how this analyzer might degrade performance
    if (m_exec_conf->getNRanks() > 1)
        {
        m_exec_conf->msg->error() << "azplugins: RDF analyzer does not support MPI" << std::endl;
        throw std::runtime_error("RDF analyzer does not support MPI");
        }
#endif // ENABLE_MPI
    }

RDFAnalyzer::~RDFAnalyzer()
    {
    m_exec_conf->msg->notice(5) << "Destroying RDFAnalyzer" << std::endl;
    }

/*!
 * \param timestep Current simulation timestep
 */
void RDFAnalyzer::analyze(unsigned int timestep)
    {
    if (m_prof) m_prof->push("RDF");

    // validate the cutoff radius in the current simulation box
    const BoxDim& box = m_pdata->getBox();
        {
        Scalar3 nearest_plane_distance = box.getNearestPlaneDistance();
        if ((box.getPeriodic().x && nearest_plane_distance.x < m_rcut * 2.0) ||
            (box.getPeriodic().y && nearest_plane_distance.y < m_rcut * 2.0) ||
            (m_sysdef->getNDimensions() == 3 && box.getPeriodic().z && nearest_plane_distance.z < m_rcut * 2.0))
            {
            m_exec_conf->msg->error() << "azplugins: Simulation box is too small to compute RDF, reduce cutoff." << std::endl;
            throw std::runtime_error("RDF analyzer cutoff radius is too large");
            }
        }

    binParticles();

    accumulate();

    if (m_prof) m_prof->pop();
    }

/*!
 * Particle pairs are binned into *m_counts* by all-pairs calculation.
 */
void RDFAnalyzer::binParticles()
    {
    if (m_prof) m_prof->push("bin");

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_group_1(m_group_1->getIndexArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_group_2(m_group_2->getIndexArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_counts(m_counts, access_location::host, access_mode::overwrite);
    memset(h_counts.data, 0, m_num_bins * sizeof(unsigned int));

    const Scalar rcutsq = m_rcut * m_rcut;
    const BoxDim& box = m_pdata->getBox();

    // compute the rdf counts for this frame
    const unsigned int N_1 = m_group_1->getNumMembers();
    const unsigned int N_2 = m_group_2->getNumMembers();
    unsigned int duplicates = 0;
    for (unsigned int i=0; i < N_1; ++i)
        {
        // load particle i
        const unsigned int idx_i = h_group_1.data[i];
        const Scalar4 postype_i = h_pos.data[idx_i];
        const Scalar3 r_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);

        for (unsigned int j=0; j < N_2; ++j)
            {
            // load particle j
            const unsigned int idx_j = h_group_2.data[j];
            if (idx_i == idx_j)
                {
                ++duplicates;
                continue;
                }

            const Scalar4 postype_j = h_pos.data[idx_j];
            const Scalar3 r_j = make_scalar3(postype_j.x, postype_j.y, postype_j.z);

            // distance calculation
            const Scalar3 dr = box.minImage(r_j - r_i);
            const Scalar drsq = dot(dr, dr);

            if (drsq < rcutsq)
                {
                const unsigned int bin = static_cast<unsigned int>(sqrt(drsq) / m_bin_width);
                ++h_counts.data[bin];
                }
            }
        }

    m_duplicates.resetFlags(duplicates);
    if (m_prof) m_prof->pop();
    }

/*!
 * To get g(r) between the groups, we take group 1 as the "origins" and group 2 as the interactions.
 * Then, the normalization should be:
 *
 * \verbatim
 *                  counts_i
 *      g_i = ----------------------
 *            pair_density * V_shell
 *
 * \endverbatim
 *
 * The pair density is pair_density = (N_1 * N_2 - duplicates)/V. When the groups are the same,
 * there are N duplicates, corresponding to the self terms, so there are N(N-1) total pairs
 * (both i,j and j,i pairs are considered). The shell volume is V_shell = (4 pi/3)*(r_{i+1}^3-r_i^3).
 */
void RDFAnalyzer::accumulate()
    {
    if (m_prof) m_prof->push("accumulate");
    ArrayHandle<unsigned int> h_counts(m_counts, access_location::host, access_mode::read);
    ArrayHandle<double> h_accum_rdf(m_accum_rdf, access_location::host, access_mode::readwrite);

    const unsigned int duplicates = m_duplicates.readFlags();
    const unsigned int N_1_global = m_group_1->getNumMembersGlobal();
    const unsigned int N_2_global = m_group_2->getNumMembersGlobal();
    const double prefactor = m_pdata->getGlobalBox().getVolume() / static_cast<double>(N_1_global * N_2_global - duplicates);

    for (unsigned int i=0; i < m_num_bins; ++i)
        {
        const double r_in = m_bin_width * static_cast<double>(i);
        const double r_out = std::min(r_in + m_bin_width, static_cast<double>(m_rcut));
        const double V_shell = (4.0*M_PI/3.0) * (r_out * r_out * r_out - r_in * r_in * r_in);
        h_accum_rdf.data[i] += static_cast<double>(h_counts.data[i]) * prefactor / V_shell;
        }
    ++m_num_samples;
    if (m_prof) m_prof->pop();
    }

/*!
 * \returns Bins distribution function was computed on
 */
std::vector<Scalar> RDFAnalyzer::getBins() const
    {
    std::vector<Scalar> bins(m_num_bins);
    for (unsigned int i=0; i < m_num_bins; ++i)
        {
        const Scalar r_in = m_bin_width * static_cast<Scalar>(i);
        const Scalar r_out = std::min(r_in + m_bin_width, m_rcut);
        bins[i] = Scalar(0.5) * (r_in + r_out);
        }
    return bins;
    }

/*!
 * \returns Accumulated radial distribution function
 */
std::vector<Scalar> RDFAnalyzer::get() const
    {
    ArrayHandle<double> h_accum_rdf(m_accum_rdf, access_location::host, access_mode::read);
    std::vector<Scalar> rdf(m_num_bins,0.0);

    if (m_num_samples > 0)
        {
        for (unsigned int i=0; i < m_num_bins; ++i)
            {
            rdf[i] = h_accum_rdf.data[i] / static_cast<double>(m_num_samples);
            }
        }
    return rdf;
    }

void RDFAnalyzer::reset()
    {
    m_num_samples = 0;
    ArrayHandle<double> h_accum_rdf(m_accum_rdf, access_location::host, access_mode::overwrite);
    memset(h_accum_rdf.data, 0, m_num_bins * sizeof(double));
    }

namespace detail
{
/*!
 * \param m Python module for export
 */
void export_RDFAnalyzer(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_< RDFAnalyzer, std::shared_ptr<RDFAnalyzer> >(m, "RDFAnalyzer", py::base<Analyzer>())
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, std::shared_ptr<ParticleGroup>, Scalar, Scalar>())
        .def("get", &RDFAnalyzer::get)
        .def("getBins", &RDFAnalyzer::getBins)
        .def("reset", &RDFAnalyzer::reset);
    }
} // end namespace detail
} // end namespace azplugins
