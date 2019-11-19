// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file VariantSphereArea.h
 * \brief Declares a variant for a shrinking / growing sphere radius.
 */

#ifndef AZPLUGINS_VARIANT_SPHERE_AREA_H_
#define AZPLUGINS_VARIANT_SPHERE_AREA_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/HOOMDMath.h"
#include "hoomd/Variant.h"

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace azplugins
{
//! Radius of sphere contracting with a constant rate of surface reduction.
/*!
 * The radius of the sphere is reduced according to the following relationship:
 *
 * \f[
 * R(t) = \sqrt{R(0)^2 - (\alpha/4\pi) t}
 * \f]
 *
 * where \f$\alpha\f$ is the rate of surface area reduction per time.
 *
 * To be physical, \f$R(t)\f$ is never negative; it is fixed to 0 if \f$t\f$ would make the
 * square-root argument negative.
 *
 * Setting \f$\alpha < 0\f$ will cause the sphere to expand.
 */
class PYBIND11_EXPORT VariantSphereArea : public Variant
    {
    public:
        //! Constructor
        VariantSphereArea(double R0, double alpha);

        //! Evaluate sphere radius
        virtual double getValue(unsigned int timestep);

    private:
        double m_R0_sq; //!< Initial radius (squared)
        double m_k;     //!< Rate of change (includes 1/4pi prefactor)
    };

namespace detail
{

//! Export VariantSphereArea to python
void export_VariantSphereArea(pybind11::module& m);

} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_VARIANT_SPHERE_AREA_H_
