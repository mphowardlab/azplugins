// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file SinusoidalChannelFiller.cc
 * \brief Definition of SinusoidalChannelFiller
 */

#include "MPCDSinusoidalChannelFiller.h"
#include "hoomd/RandomNumbers.h"
#include "RNGIdentifiers.h"

namespace azplugins
{
SinusoidalChannelFiller::SinusoidalChannelFiller(std::shared_ptr<mpcd::SystemData> sysdata,
                                                 Scalar density,
                                                 unsigned int type,
                                                 std::shared_ptr<::Variant> T,
                                                 unsigned int seed,
                                                 std::shared_ptr<const detail::SinusoidalChannel> geom)
    : mpcd::VirtualParticleFiller(sysdata, density, type, T, seed), m_geom(geom)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD SinusoidalChannelFiller" << std::endl;
    }

SinusoidalChannelFiller::~SinusoidalChannelFiller()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD SinusoidalChannelFiller" << std::endl;
    }

void SinusoidalChannelFiller::computeNumFill()
    {
    // as a precaution, validate the global box with the current cell list
    const BoxDim& global_box = m_pdata->getGlobalBox();
    const Scalar cell_size = m_cl->getCellSize();
    const Scalar max_shift = m_cl->getMaxGridShift();
    if (!m_geom->validateBox(global_box, cell_size))
        {
        m_exec_conf->msg->error() << "Invalid cosine geometry for global box, cannot fill virtual particles." << std::endl;
        m_exec_conf->msg->error() << "Filler thickness is given by cell_size +  0.5*A*sin((cell_size+max_shift)*2*pi*p/L); " << std::endl;
        throw std::runtime_error("Invalid AntiSymCos geometry for global box");
        }

    // default is not to fill anything
    m_thickness = 0;
    m_N_fill = 0;
    m_pi_period_div_L = 0;
    m_Amplitude = 0;
    m_H_narrow = 0;

    // box and sine geometry
    const BoxDim& box = m_pdata->getBox();
    const Scalar3 L = box.getL();
    const Scalar Area = L.x * L.y;
    const Scalar A = m_geom->getAmplitude();
    const Scalar h = m_geom->getHnarrow();
    const Scalar r = m_geom->getRepetitions();
    m_Amplitude=A;
    m_pi_period_div_L = 2*M_PI*r/L.x;
    m_H_narrow = h;

    // This geometry needs a larger filler thickness than just a single cell_size because of its curved bounds.
    const Scalar filler_thickness = cell_size +  m_Amplitude*fast::sin((cell_size+max_shift)*m_pi_period_div_L);
    m_thickness = filler_thickness;
    // total number of fill particles
    m_N_fill = m_density*Area*filler_thickness*2;
    }

/*!
 * \param timestep Current timestep to draw particles
 */
void SinusoidalChannelFiller::drawParticles(unsigned int timestep)
    {
    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(), access_location::host, access_mode::readwrite);

    const BoxDim& box = m_pdata->getBox();
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();
    const unsigned int N_half = m_N_fill/2;
    const Scalar vel_factor = fast::sqrt(m_T->getValue(timestep) / m_mpcd_pdata->getMass());

    // index to start filling from
    const unsigned int first_idx = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual() - m_N_fill;

    for (unsigned int i=0; i < m_N_fill; ++i)
        {
        const unsigned int tag = m_first_tag + i;
        hoomd::RandomGenerator rng(RNGIdentifier::SinusoidalChannelFiller, m_seed, tag, timestep);
        signed char sign = (i >= N_half) - (i < N_half); // bottom -1 or top +1

        Scalar x = hoomd::UniformDistribution<Scalar>(lo.x, hi.x)(rng);
        Scalar y = hoomd::UniformDistribution<Scalar>(lo.y, hi.y)(rng);
        Scalar z = hoomd::UniformDistribution<Scalar>(0, sign*m_thickness)(rng);

        z = m_Amplitude*fast::cos(x*m_pi_period_div_L)+sign*m_H_narrow + z;

        const unsigned int pidx = first_idx + i;
        h_pos.data[pidx] = make_scalar4(x,
                                        y,
                                        z,
                                        __int_as_scalar(m_type));

        //m_exec_conf->msg->notice(0)<< "filler " << x << " "<< y << " "<< z << std::endl;

        hoomd::NormalDistribution<Scalar> gen(vel_factor, 0.0);
        Scalar3 vel;
        gen(vel.x, vel.y, rng);
        vel.z = gen(rng);
        // TODO: should these be given zero net-momentum contribution (relative to the frame of reference?)
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
void export_SinusoidalChannelFiller(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<SinusoidalChannelFiller, std::shared_ptr<SinusoidalChannelFiller>>
        (m, "SinusoidalChannelFiller", py::base<mpcd::VirtualParticleFiller>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>,
             Scalar,
             unsigned int,
             std::shared_ptr<::Variant>,
             unsigned int,
             std::shared_ptr<const SinusoidalChannel>>())
        .def("setGeometry", &SinusoidalChannelFiller::setGeometry)
        ;
    }

} // end namespace detail

} // end namespace azplugins
