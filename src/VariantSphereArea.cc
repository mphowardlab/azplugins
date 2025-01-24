// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file VariantSphereArea.cc
 * \brief Defines a variant for a shrinking / growing sphere radius.
 */

#include "VariantSphereArea.h"

namespace azplugins
    {
/*!
 * \param R0 Initial radius.
 * \param alpha Rate of surface reduction (units are area per timestep).
 */
VariantSphereArea::VariantSphereArea(double R0, double alpha)
    {
    m_R0_sq = R0 * R0;
    m_k = alpha / (4. * M_PI);
    }

/*!
 * \param timestep Current simulation timestep.
 * \returns Current radius of sphere.
 */
double VariantSphereArea::getValue(unsigned int timestep)
    {
    const double drsq = m_k * timestep;

    // droplet cannot shrink smaller than zero
    if (drsq >= m_R0_sq)
        {
        return 0.;
        }
    else
        {
        return slow::sqrt(m_R0_sq - drsq);
        }
    }

namespace detail
    {
/*!
 * \param m Python module to export to.
 */
void export_VariantSphereArea(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<VariantSphereArea, std::shared_ptr<VariantSphereArea>>(m,
                                                                      "VariantSphereArea",
                                                                      py::base<Variant>())
        .def(py::init<double, double>());
    }

    } // end namespace detail
    } // end namespace azplugins
