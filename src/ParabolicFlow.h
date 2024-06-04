// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file ParabolicFlow.h
 * \brief Declaration of ParabolicFlow
 */

#ifndef AZPLUGINS_PARABOLIC_FLOW_H_
#define AZPLUGINS_PARABOLIC_FLOW_H_

#include "hoomd/HOOMDMath.h"

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#include <pybind11/pybind11.h>
#endif
namespace hoomd
    {
namespace azplugins
    {
class ParabolicFlow
    {
    public:
    //! Construct parabolic flow profile
    /*!
     * \param U_ Mean velocity
     * \param L_ Separation
     */
    ParabolicFlow(Scalar U_, Scalar L_) : Umax(Scalar(1.5) * U_), L(Scalar(0.5) * L_) { }

    //! Evaluate the flow field
    /*!
     * \param r position to evaluate flow
     */
    HOSTDEVICE Scalar3 operator()(const Scalar3& r) const
        {
        const Scalar zr = (r.z / L);
        return make_scalar3(Umax * (1. - zr * zr), 0.0, 0.0);
        }

    HOSTDEVICE Scalar getVelocity() const
        {
        return Scalar(0.6666666667) * Umax;
        }

    HOSTDEVICE void setVelocity(const Scalar& U)
        {
        Umax = Scalar(1.5) * U;
        }

    HOSTDEVICE Scalar getLength() const
        {
        return Scalar(2.0) * L;
        }

    HOSTDEVICE void setLength(const Scalar& L_)
        {
        L = Scalar(0.5) * L_;
        }

    private:
    Scalar Umax; //<! Mean velocity
    Scalar L;    //!< Full width
    };

namespace detail
    {
void export_ParabolicFlow(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<ParabolicFlow, std::shared_ptr<ParabolicFlow>>(m, "ParabolicFlow")
        .def(py::init<Scalar, Scalar>())
        .def_property("mean_velocity", &ParabolicFlow::getVelocity, &ParabolicFlow::setVelocity)
        .def_property("separation", &ParabolicFlow::getLength, &ParabolicFlow::setLength);
    }
    } // namespace detail
    } // namespace azplugins
    } // namespace hoomd
#undef HOSTDEVICE

#endif // AZPLUGINS_PARABOLIC_FLOW_H_
