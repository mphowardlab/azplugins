// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file ParabolicFlow.h
 * \brief Declaration of ParabolicFlow
 */

#ifndef AZPLUGINS_PARABOLIC_FLOW_H_
#define AZPLUGINS_PARABOLIC_FLOW_H_

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

#include "hoomd/HOOMDMath.h"

#ifndef __HIPCC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif // __HIPCC__

#ifndef PYBIND11_EXPORT
#define PYBIND11_EXPORT __attribute__((visibility("default")))
#endif

namespace hoomd
    {
namespace azplugins
    {

//! Unidirectional parabolic flow field
/*!
 * 1d flow along the \a x axis. The geometry is a parallel plate channel with
 * the plates centered around \f$ y = 0 \f$ and positioned at \f$ \pm L \f$.
 * The \a y axis is the vorticity direction and periodic. The flow profile in
 * this geometry is then
 *
 * \f[
 * u_x(y) = \frac{3}{2} U \left[1 - \left(\frac{y}{L}\right)^2 \right]
 * \f]
 *
 * Here, \f$ mean_velocity \f$ is the mean velocity, which is related to the pressure drop
 * and viscosity.
 *
 * \note The user must properly establish no flux of particles through the channel
 *       walls through an appropriate wall potential.
 */
class PYBIND11_EXPORT ParabolicFlow
    {
    public:
    //! Construct parabolic flow profile
    /*!
     * \param U_ Mean velocity
     * \param L_ Separation
     */
    ParabolicFlow(Scalar mean_velocity, Scalar separation)
        {
        setMeanVelocity(mean_velocity);
        setSeparation(separation);
        }

    //! Evaluate the flow field
    /*!
     * \param r position to evaluate flow
     */
    HOSTDEVICE Scalar3 operator()(const Scalar3& r) const
        {
        const Scalar yr = (r.y / L);
        return make_scalar3(Umax * (1. - yr * yr), 0.0, 0.0);
        }

    HOSTDEVICE Scalar getMeanVelocity() const
        {
        return Umax / Scalar(1.5);
        }

    HOSTDEVICE void setMeanVelocity(const Scalar& U)
        {
        Umax = Scalar(1.5) * U;
        }

    HOSTDEVICE Scalar getSeparation() const
        {
        return Scalar(2.0) * L;
        }

    HOSTDEVICE void setSeparation(const Scalar& L_)
        {
        L = Scalar(0.5) * L_;
        }

    private:
    Scalar Umax; //<! Mean velocity
    Scalar L;    //!< Full width
    };

    } // namespace azplugins
    } // namespace hoomd

#undef HOSTDEVICE
#undef PYBIND11_EXPORT

#endif // AZPLUGINS_PARABOLIC_FLOW_H_
