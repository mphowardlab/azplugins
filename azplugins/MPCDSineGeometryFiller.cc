// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file mpcd/SlitGeometryFiller.cc
 * \brief Definition of mpcd::SlitGeometryFiller
 */

#include "MPCDSineGeometryFiller.h"
#include "hoomd/RandomNumbers.h"
#include "RNGIdentifiers.h"

namespace azplugins
{
SineGeometryFiller::SineGeometryFiller(std::shared_ptr<mpcd::SystemData> sysdata,
                                             Scalar density,
                                             unsigned int type,
                                             std::shared_ptr<::Variant> T,
                                             unsigned int seed,
                                             std::shared_ptr<const detail::SineGeometry> geom)
    : mpcd::VirtualParticleFiller(sysdata, density, type, T, seed), m_geom(geom)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD SineGeometryFiller" << std::endl;
    }

SineGeometryFiller::~SineGeometryFiller()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD SineGeometryFiller" << std::endl;
    }

void SineGeometryFiller::computeNumFill()
    {
    // as a precaution, validate the global box with the current cell list
    const BoxDim& global_box = m_pdata->getGlobalBox();
    const Scalar cell_size = m_cl->getCellSize();
    const Scalar max_shift = m_cl->getMaxGridShift();
    if (!m_geom->validateBox(global_box, cell_size))
        {
        m_exec_conf->msg->error() << "Invalid sine geometry for global box, cannot fill virtual particles." << std::endl;
        m_exec_conf->msg->error() << "Filler thickness is given by cell_size +  0.5*(H-h)*sin((cell_size+max_shift)*2*pi*p/L); " << std::endl;
        throw std::runtime_error("Invalid sine geometry for global box");
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

    // This geometry needs a larger filler thickness than just a single cell_size because of its curved bounds.
    const Scalar filler_thickness = cell_size +  m_amplitude*fast::sin((cell_size+max_shift)*m_pi_period_div_L);
    m_thickness = filler_thickness;
    // total number of fill particles
    m_N_fill = m_density*Area*filler_thickness*2;
    }

/*!
 * \param timestep Current timestep to draw particles
 */
void SineGeometryFiller::drawParticles(unsigned int timestep)
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
        hoomd::RandomGenerator rng(RNGIdentifier::SineGeometryFiller, m_seed, tag, timestep);
        signed char sign = (i >= N_half) - (i < N_half); // bottom -1 or top +1

        Scalar x = hoomd::UniformDistribution<Scalar>(lo.x, hi.x)(rng);
        Scalar y = hoomd::UniformDistribution<Scalar>(lo.y, hi.y)(rng);
        Scalar z = hoomd::UniformDistribution<Scalar>(0, sign*m_thickness)(rng);

        z = sign*(m_amplitude*fast::cos(x*m_pi_period_div_L)+m_amplitude + m_H_narrow ) + z;

        const unsigned int pidx = first_idx + i;
        h_pos.data[pidx] = make_scalar4(x,
                                        y,
                                        z,
                                        __int_as_scalar(m_type));

        //m_exec_conf->msg->notice(5) << x << " "<< y << " "<< z << std::endl;

        hoomd::NormalDistribution<Scalar> gen(vel_factor, 0.0);
        Scalar3 vel;
        gen(vel.x, vel.y, rng);
        vel.z = gen(rng);
        // TODO: should these be given zero net-momentum contribution (relative to the frame of reference?)
        h_vel.data[pidx] = make_scalar4(vel.x + sign * m_geom->getVelocity(),
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
void export_SineGeometryFiller(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<SineGeometryFiller, std::shared_ptr<SineGeometryFiller>>
        (m, "SineGeometryFiller", py::base<mpcd::VirtualParticleFiller>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>,
                      Scalar,
                      unsigned int,
                      std::shared_ptr<::Variant>,
                      unsigned int,
                      std::shared_ptr<const SineGeometry>>())
        .def("setGeometry", &SineGeometryFiller::setGeometry)
        ;
    }

} // end namespace detail

} // end namespace azplugins
