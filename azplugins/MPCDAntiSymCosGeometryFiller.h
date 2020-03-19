// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file AntiSymCosGeometryFiller.h
 * \brief Definition of virtual particle filler for mpcd::detail::AntiSymCosGeometry.
 */

#ifndef AZPLUGINS_MPCD_ANTI_SYM_COS_GEOMETRY_FILLER_H_
#define AZPLUGINS_MPCD_ANTI_SYM_COS_GEOMETRY_FILLER_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "MPCDAntiSymCosGeometry.h"
#include "hoomd/mpcd/VirtualParticleFiller.h"
#include "hoomd/mpcd/SystemData.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace azplugins
{

//! Adds virtual particles to the MPCD particle data for AntiSymCosGeometry
/*!
 * Particles are added to the volume that is overlapped by any of the cells that are also "inside" the channel,
 * subject to the grid shift.
 */
class PYBIND11_EXPORT AntiSymCosGeometryFiller : public mpcd::VirtualParticleFiller
    {
    public:
        AntiSymCosGeometryFiller(std::shared_ptr<mpcd::SystemData> sysdata,
                           Scalar density,
                           unsigned int type,
                           std::shared_ptr<::Variant> T,
                           unsigned int seed,
                           std::shared_ptr<const detail::AntiSymCosGeometry> geom);

        virtual ~AntiSymCosGeometryFiller();

        void setGeometry(std::shared_ptr<const detail::AntiSymCosGeometry> geom)
            {
            m_geom = geom;
            }

    protected:
        std::shared_ptr<const detail::AntiSymCosGeometry> m_geom;
        Scalar m_thickness;       //!< thickness of virtual particle buffer zone
        Scalar m_Amplitude;       //!< amplitude of  channel wall cosine: 0.5(H_wide - H_narrow)
        Scalar m_pi_period_div_L; //!< period of channel wall cosine: 2*pi*period/Lx
        Scalar m_H_narrow;        //!< half width of the narrowest height of the channel

        //! Compute the total number of particles to fill
        virtual void computeNumFill();

        //! Draw particles within the fill volume
        virtual void drawParticles(unsigned int timestep);
    };

namespace detail
{
//! Export AntiSymCosGeometryFiller to python
void export_AntiSymCosGeometryFiller(pybind11::module& m);
} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_MPCD_ANTI_SYM_COS_GEOMETRY_FILLER_H_
