// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_PLANAR_BARRIER_EVALUATOR_H_
#define AZPLUGINS_PLANAR_BARRIER_EVALUATOR_H_

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif // __HIPCC__

namespace hoomd
    {
namespace azplugins
    {

//! Planar barrier
/*!
 * The barrier is a plane with normal in the +y direction. The harmonic potential
 * acts to move particles that are above y = H down, with H optionally being
 * shifted by an offset. The plane must lie inside the axis-aligned bounding
 * box enclosing the global simulation box. The coordinates are assumed to be
 * wrapped into the global box.
 */
class PlanarBarrierEvaluator
    {
    public:
    HOSTDEVICE PlanarBarrierEvaluator(Scalar H) : m_H(H) { }

    //! Evaluate the harmonic barrier force and energy
    HOSTDEVICE Scalar4 operator()(const Scalar3& pos, Scalar k, Scalar offset) const
        {
        const Scalar dy = pos.y - (m_H + offset);
        // don't compute if below minimum of potential
        if (dy <= Scalar(0.0))
            {
            return make_scalar4(0, 0, 0, 0);
            }

        const Scalar f = -k * dy;               // y component of force
        const Scalar e = Scalar(-0.5) * f * dy; // U = 0.5 k dx^2
        return make_scalar4(0, f, 0, e);
        }

    //! Validate the harmonic barrier location is inside the global box
    HOSTDEVICE bool valid(const BoxDim& box) const
        {
        const Scalar3 lo = box.makeCoordinates(make_scalar3(0, 0, 0));
        const Scalar3 hi = box.makeCoordinates(make_scalar3(1, 1, 1));
        return (m_H >= lo.y && m_H < hi.y);
        }

    private:
    Scalar m_H; //<! y position of interface
    };

    } // end namespace azplugins
    } // end namespace hoomd

#undef HOSTDEVICE

#endif // AZPLUGINS_PLANAR_BARRIER_EVALUATOR_H_
