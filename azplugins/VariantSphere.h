// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file VariantSphere.h
 * \brief Declares a variant for a shrinking / growing sphere radius.
 */

#ifndef AZPLUGINS_VARIANT_SPHERE_H_
#define AZPLUGINS_VARIANT_SPHERE_H_

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
 * The volume of the sphere is reduced according to the following relationship:
 *
 * \f[
 * V(t)^{2/3} = V(0)^{2/3} - \alpha t
 * \f]
 *
 * where the sphere volume is \f$V(t) = 4\piR(t)^3/3$. The droplet radius is then given by
 *
 * \f[
 * R(t) = \sqrt{R(0)^2 - (1/4)(6/\pi)^{2/3} \alpha t}
 * \f]
 *
 * To be physical, \f$R(t)\f$ is never negative; it is fixed to 0 if \f$t\f$ would make the
 * square-root argument negative.
 *
 * Setting \f$\alpha < 0\f$ will cause the sphere to expand.
 */
class PYBIND11_EXPORT VariantSphere : public Variant
    {
    public:
        //! Constructor
        VariantSphere(double R0, double alpha);

        //! Evaluate sphere radius
        virtual double getValue(unsigned int timestep);

    private:
        double m_R0_sq; //!< Initial radius (squared)
        double m_k;     //!< Rate of change (includes constant prefactors)
    };

namespace detail
{

//! Export VariantSphere to python
void export_VariantSphere(pybind11::module& m);

} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_VARIANT_SPHERE_H_
