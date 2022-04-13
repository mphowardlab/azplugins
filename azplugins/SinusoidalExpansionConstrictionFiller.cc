// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file SinusoidalExpansionConstrictionFiller.cc
 * \brief Definition of SinusoidalExpansionConstrictionFiller
 */

#include "SinusoidalExpansionConstrictionFiller.h"
#include "hoomd/RandomNumbers.h"
#include "RNGIdentifiers.h"

namespace azplugins
{
SinusoidalExpansionConstrictionFiller::SinusoidalExpansionConstrictionFiller(std::shared_ptr<mpcd::SystemData> sysdata,
                                                                             Scalar density,
                                                                             unsigned int type,
                                                                             std::shared_ptr<::Variant> T,
                                                                             unsigned int seed,
                                                                             std::shared_ptr<const detail::SinusoidalExpansionConstriction> geom)
        : mpcd::VirtualParticleFiller(sysdata, density, type, T, seed), m_geom(geom)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD SinusoidalExpansionConstrictionFiller" << std::endl;
    }

SinusoidalExpansionConstrictionFiller::~SinusoidalExpansionConstrictionFiller()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD SinusoidalExpansionConstrictionFiller" << std::endl;
    }

void SinusoidalExpansionConstrictionFiller::computeNumFill()
    {
    // as a precaution, validate the global box with the current cell list
    const BoxDim& global_box = m_pdata->getGlobalBox();
    const Scalar cell_size = m_cl->getCellSize();
    const Scalar max_shift = m_cl->getMaxGridShift();
    if (!m_geom->validateBox(global_box, cell_size))
        {
        m_exec_conf->msg->error() << "Invalid cosine geometry for global box, cannot fill virtual particles." << std::endl;
        m_exec_conf->msg->error() << "Filler thickness is given by cell_size +  0.5*(H-h)*sin((cell_size+max_shift)*2*pi*p/L); " << std::endl;
        throw std::runtime_error("Invalid SinsuodialExpansionConstriction geometry for global box");
        }

    // default is not to fill anything
    m_thickness = 0;
    m_N_fill = 0;
    m_pi_period_div_L = 0;
    m_amplitude = 0;

    // box and sine geometry
    const BoxDim& box = m_pdata->getBox();
    const Scalar3 L = box.getL();
    const Scalar Area = L.x * L.y;
    const Scalar H = m_geom->getHwide();
    const Scalar h = m_geom->getHnarrow();
    const Scalar r = m_geom->getRepetitions();

    m_amplitude = 0.5*(H-h);
    m_pi_period_div_L = 2*M_PI*r/L.x;
    m_H_narrow = h;

    /* This geometry needs a larger filler thickness than just a single cell_size because of its curved bounds.
     * Each cell along the wall boundary must be filled such that a cell shifted with max_shift is still entirely covered.
     * At the top/bottom this is straightforward, so cell_size+max_shift should be enough (max_shift = 0.5*cell_size).
     * At the steepest point of the sine (i.e., around the zero crossing), we actually need more thickness, such that the
     * diagonal of cell_size+max_shift fits into the filler area. The equation below comes from calculating the minimum
     * shift necessary to fit the diagonal of a by max_shift shifted cell at the narrowest point.
     * It will fail when A=0 (but then the slit geometry should be used anyway). This creates a filler that is at least
     * cell_size+max_shift wide everywhere.
     */
    m_thickness = cell_size +  m_amplitude*fast::sin((cell_size+max_shift)*m_pi_period_div_L);
    m_N_fill = 2*m_density*Area*m_thickness;
    }

/*!
 * \param timestep Current timestep to draw particles
 */
void SinusoidalExpansionConstrictionFiller::drawParticles(unsigned int timestep)
    {
    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(), access_location::host, access_mode::readwrite);

    const BoxDim& box = m_pdata->getBox();
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();
    const unsigned int N_half = 0.5*m_N_fill;
    const Scalar vel_factor = fast::sqrt(m_T->getValue(timestep) / m_mpcd_pdata->getMass());

    // index to start filling from
    const unsigned int first_idx = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual() - m_N_fill;

    for (unsigned int i=0; i < m_N_fill; ++i)
        {
        const unsigned int tag = m_first_tag + i;
        hoomd::RandomGenerator rng(RNGIdentifier::SinusoidalExpansionConstrictionFiller, m_seed, tag, timestep);
        signed char sign = (i >= N_half) - (i < N_half); // bottom -1 or top +1

        Scalar x = hoomd::UniformDistribution<Scalar>(lo.x, hi.x)(rng);
        Scalar y = hoomd::UniformDistribution<Scalar>(lo.y, hi.y)(rng);
        Scalar z = hoomd::UniformDistribution<Scalar>(0, sign*m_thickness)(rng);
        z += sign*(m_amplitude*fast::cos(x*m_pi_period_div_L)+m_amplitude + m_H_narrow);

        const unsigned int pidx = first_idx + i;
        h_pos.data[pidx] = make_scalar4(x, y, z, __int_as_scalar(m_type));

        hoomd::NormalDistribution<Scalar> gen(vel_factor, 0.0);
        Scalar3 vel;
        gen(vel.x, vel.y, rng);
        vel.z = gen(rng);

        h_vel.data[pidx] = make_scalar4(vel.x,
                                        vel.y,
                                        vel.z,
                                        __int_as_scalar(mpcd::detail::NO_CELL));
        h_tag.data[pidx] = tag;
        }
    }

namespace detail
{
/*!
 * \param m Python module to export to
 */
void export_SinusoidalExpansionConstrictionFiller(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<SinusoidalExpansionConstrictionFiller, std::shared_ptr<SinusoidalExpansionConstrictionFiller>>
        (m, "SinusoidalExpansionConstrictionFiller", py::base<mpcd::VirtualParticleFiller>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>,
                      Scalar,
                      unsigned int,
                      std::shared_ptr<::Variant>,
                      unsigned int,
                      std::shared_ptr<const SinusoidalExpansionConstriction>>())
        .def("setGeometry", &SinusoidalExpansionConstrictionFiller::setGeometry)
        ;
    }

} // end namespace detail

} // end namespace azplugins
