// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file VariantSphere.cc
 * \brief Defines a variant for a shrinking / growing sphere radius.
 */

#include "VariantSphere.h"

namespace azplugins
{
/*!
 * \param R0 Initial radius.
 * \param alpha Rate of surface reduction (units are per timestep).
 *
 * The parameters are converted into quantities that are more useful internally.
 * The units of alpha are implicitly per-timestep.
 */
VariantSphere::VariantSphere(double R0, double alpha)
    {
    // precompute quantities for easy evaluation later
    m_R0_sq = R0*R0;

    const Scalar factor = 0.3848347315591266;   // (6/pi)^(2/3) / 4
    m_k = factor*alpha;
    }

/*!
 * \param timestep Current simulation timestep.
 * \returns Current radius of sphere.
 */
double VariantSphere::getValue(unsigned int timestep)
    {
    const double drsq = m_k*timestep;

    // droplet cannot shrink smaller than zero
    if (drsq >= m_R0_sq)
        {
        return 0.;
        }
    else
        {
        return slow::sqrt(m_R0_sq-drsq);
        }
    }

namespace detail
{
/*!
 * \param m Python module to export to.
 */
void export_VariantSphere(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<VariantSphere, std::shared_ptr<VariantSphere>>(m,"VariantSphere",py::base<Variant>())
    .def(py::init<double,double>());
    }

} // end namespace detail
} // end namespace azplugins
