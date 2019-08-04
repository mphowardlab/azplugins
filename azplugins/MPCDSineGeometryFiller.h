// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file mpcd/SlitGeometryFiller.h
 * \brief Definition of virtual particle filler for mpcd::detail::SlitGeometry.
 */

#ifndef AZPLUGINS_MPCD_SINE_GEOMETRY_FILLER_H_
#define AZPLUGINS_MPCD_SINE_GEOMETRY_FILLER_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "MPCDSineGeometry.h"
#include "hoomd/mpcd/VirtualParticleFiller.h"
#include "hoomd/mpcd/SystemData.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace azplugins
{

//! Adds virtual particles to the MPCD particle data for SlitGeometry
/*!
 * Particles are added to the volume that is overlapped by any of the cells that are also "inside" the channel,
 * subject to the grid shift.
 */
class PYBIND11_EXPORT SineGeometryFiller : public mpcd::VirtualParticleFiller
    {
    public:
        SineGeometryFiller(std::shared_ptr<mpcd::SystemData> sysdata,
                           Scalar density,
                           unsigned int type,
                           std::shared_ptr<::Variant> T,
                           unsigned int seed,
                           std::shared_ptr<const detail::SineGeometry> geom);

        virtual ~SineGeometryFiller();

        void setGeometry(std::shared_ptr<const detail::SineGeometry> geom)
            {
            m_geom = geom;
            }

    protected:
        std::shared_ptr<const detail::SineGeometry> m_geom;
        Scalar m_z_min; //!< Min z coordinate for filling
        Scalar m_z_max; //!< Max z coordinate for filling
        unsigned int m_N_lo;    //!< Number of particles to fill below channel
        unsigned int m_N_hi;    //!< number of particles to fill above channel

        //! Compute the total number of particles to fill
        virtual void computeNumFill();

        //! Draw particles within the fill volume
        virtual void drawParticles(unsigned int timestep);
    };

namespace detail
{
//! Export SlitGeometryFiller to python
void export_SineGeometryFiller(pybind11::module& m);
} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_MPCD_SINE_GEOMETRY_FILLER_H_
